/*
 * Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <cassert>
#include <vector>
#include "NvInfer.h"
#include "NvUffParser.h"

#include "NvUtils.h"
#include "argsParser.h"

using namespace nvuffparser;
using namespace nvinfer1;

#include "logger.h"
#include "common.h"

const std::string gSampleName = "TensorRT.sample_uff_mnist";

samplesCommon::Args gArgs;

#define MAX_WORKSPACE (1 << 30)

inline int64_t volume(const Dims& d)
{
    int64_t v = 1;
    for (int64_t i = 0; i < d.nbDims; i++)
        v *= d.d[i];
    return v;
}

inline unsigned int elementSize(DataType t)
{
    switch (t)
    {
    case DataType::kINT32:
        // Fallthrough, same as kFLOAT
    case DataType::kFLOAT: return 4;
    case DataType::kHALF: return 2;
    case DataType::kINT8: return 1;
    }
    assert(0);
    return 0;
}

static const int INPUT_H = 28;
static const int INPUT_W = 28;

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(locateFile(filename, gArgs.dataDirs), buffer, INPUT_H, INPUT_W);
}

void* safeCudaMalloc(size_t memSize)
{
    void* deviceMem;
    CHECK(cudaMalloc(&deviceMem, memSize));
    if (deviceMem == nullptr)
    {
        std::cerr << "Out of memory" << std::endl;
        exit(1);
    }
    return deviceMem;
}

std::vector<std::pair<int64_t, DataType>>
calculateBindingBufferSizes(const ICudaEngine& engine, int nbBindings, int batchSize)
{
    std::vector<std::pair<int64_t, DataType>> sizes;
    for (int i = 0; i < nbBindings; ++i)
    {
        Dims dims = engine.getBindingDimensions(i);
        DataType dtype = engine.getBindingDataType(i);

        int64_t eltCount = volume(dims) * batchSize;
        sizes.push_back(std::make_pair(eltCount, dtype));
    }

    return sizes;
}

void* createMnistCudaBuffer(int64_t eltCount, DataType dtype, int num)
{
    /* in that specific case, eltCount == INPUT_H * INPUT_W */
    assert(eltCount == INPUT_H * INPUT_W);
    assert(elementSize(dtype) == sizeof(float));

    size_t memSize = eltCount * elementSize(dtype);
    float* inputs = new float[eltCount];

    /* read PGM file */
    uint8_t fileData[INPUT_H * INPUT_W];
    readPGMFile(std::to_string(num) + ".pgm", fileData);

    /* display the number in an ascii representation */
    gLogInfo << "Input:\n";
    for (int i = 0; i < eltCount; i++)
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
    gLogInfo << std::endl;

    /* initialize the inputs buffer */
    for (int i = 0; i < eltCount; i++)
        inputs[i] = 1.0 - float(fileData[i]) / 255.0;

    void* deviceMem = safeCudaMalloc(memSize);
    CHECK(cudaMemcpy(deviceMem, inputs, memSize, cudaMemcpyHostToDevice));

    delete[] inputs;
    return deviceMem;
}

bool verifyOutput(int64_t eltCount, DataType dtype, void* buffer, int num)
{
    assert(elementSize(dtype) == sizeof(float));

    bool pass = false;

    size_t memSize = eltCount * elementSize(dtype);
    float* outputs = new float[eltCount];
    CHECK(cudaMemcpy(outputs, buffer, memSize, cudaMemcpyDeviceToHost));

    int maxIdx = 0;
    for (int i = 0; i < eltCount; ++i)
        if (outputs[i] > outputs[maxIdx])
            maxIdx = i;

    std::ios::fmtflags prevSettings = gLogInfo.flags();
    gLogInfo.setf(std::ios::fixed, std::ios::floatfield);
    gLogInfo.precision(6);
    gLogInfo << "Output:\n";
    for (int64_t eltIdx = 0; eltIdx < eltCount; ++eltIdx)
    {
        gLogInfo << eltIdx << " => " << setw(10) << outputs[eltIdx] << "\t : ";
        if (eltIdx == maxIdx)
        {
            gLogInfo << "***";
            if (eltIdx == num)
                pass = true;
        }
        gLogInfo << "\n";
    }
    gLogInfo.flags(prevSettings);

    gLogInfo << std::endl;

    delete[] outputs;

    return pass;
}

ICudaEngine* loadModelAndCreateEngine(const char* uffFile, int maxBatchSize,
                                      IUffParser* parser)
{
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);
    INetworkDefinition* network = builder->createNetwork();

#if 1
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kFLOAT))
    {
        gLogError << "Failure while parsing UFF file" << std::endl;
        return nullptr;
    }
#else
    if (!parser->parse(uffFile, *network, nvinfer1::DataType::kHALF))
    {
        gLogError << "Failure while parsing UFF file" << std::endl;
        return nullptr;
    }
    builder->setFp16Mode(true);
#endif

    /* we create the engine */
    builder->setMaxBatchSize(maxBatchSize);
    builder->setMaxWorkspaceSize(MAX_WORKSPACE);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);
    
    if (gArgs.runInInt8)
    {
        samplesCommon::setAllTensorScales(network, 127.0f, 127.0f);
    }

    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (!engine)
    {
        gLogError << "Unable to create engine" << std::endl;
        return nullptr;
    }

    /* we can clean the network and the parser */
    network->destroy();
    builder->destroy();

    return engine;
}

bool execute(ICudaEngine& engine)
{
    IExecutionContext* context = engine.createExecutionContext();

    int batchSize = 1;

    int nbBindings = engine.getNbBindings();
    assert(nbBindings == 2);

    std::vector<void*> buffers(nbBindings);
    auto buffersSizes = calculateBindingBufferSizes(engine, nbBindings, batchSize);

    int bindingIdxInput = 0;
    for (int i = 0; i < nbBindings; ++i)
    {
        if (engine.bindingIsInput(i))
            bindingIdxInput = i;
        else
        {
            auto bufferSizesOutput = buffersSizes[i];
            buffers[i] = safeCudaMalloc(bufferSizesOutput.first * elementSize(bufferSizesOutput.second));
        }
    }

    auto bufferSizesInput = buffersSizes[bindingIdxInput];

    int iterations = 1;
    int numberRun = 10;
    bool pass = true;
    for (int i = 0; i < iterations; i++)
    {
        float total = 0, ms;
        for (int num = 0; num < numberRun; num++)
        {
            buffers[bindingIdxInput] = createMnistCudaBuffer(bufferSizesInput.first,
                                                             bufferSizesInput.second, num);

            auto t_start = std::chrono::high_resolution_clock::now();
            context->execute(batchSize, &buffers[0]);
            auto t_end = std::chrono::high_resolution_clock::now();
            ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
            total += ms;

            for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
            {
                if (engine.bindingIsInput(bindingIdx))
                    continue;

                auto bufferSizesOutput = buffersSizes[bindingIdx];
                pass &= verifyOutput(bufferSizesOutput.first, bufferSizesOutput.second,
                            buffers[bindingIdx], num);
            }
            CHECK(cudaFree(buffers[bindingIdxInput]));
        }

        total /= numberRun;
        gLogInfo << "Average over " << numberRun << " runs is " << total << " ms." << std::endl;
    }

    for (int bindingIdx = 0; bindingIdx < nbBindings; ++bindingIdx)
        if (!engine.bindingIsInput(bindingIdx))
            CHECK(cudaFree(buffers[bindingIdx]));
    context->destroy();
    return pass;
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_uff_mnist [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


int main(int argc, char** argv)
{
    bool argsOK = samplesCommon::parseArgs(gArgs, argc, argv);
    if (gArgs.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }
    if (!argsOK)
    {
        gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (gArgs.dataDirs.empty())
    {
        gArgs.dataDirs = std::vector<std::string>{"data/samples/mnist/", "data/mnist/"};
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    auto fileName = locateFile("lenet5.uff", gArgs.dataDirs);
    gLogInfo << fileName << std::endl;

    int maxBatchSize = 1;
    auto parser = createUffParser();

    /* Register tensorflow input */
    parser->registerInput("in", Dims3(1, 28, 28), UffInputOrder::kNCHW);
    parser->registerOutput("out");

    ICudaEngine* engine = loadModelAndCreateEngine(fileName.c_str(), maxBatchSize, parser);

    if (!engine)
    {
        gLogError << "Model load failed" << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    /* we need to keep the memory created by the parser */
    parser->destroy();

    bool pass = execute(*engine);
    engine->destroy();
    shutdownProtobufLibrary();

    return gLogger.reportTest(sampleTest, pass);
}
