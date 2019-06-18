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

#include "NvInfer.h"
#include "logger.h"
#include "common.h"
#include "argsParser.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cuda_profiler_api.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <sys/stat.h>
#include <vector>

// constants that are known about the MNIST MLP network.
static const int32_t INPUT_H{28};                                                    // The height of the mnist input image.
static const int32_t INPUT_W{28};                                                    // The weight of the mnist input image.
static const int32_t HIDDEN_COUNT{2};                                                // The number of hidden layers for MNIST sample.
static const int32_t HIDDEN_SIZE{256};                                               // The size of the hidden state for MNIST sample.
static const int32_t MAX_BATCH_SIZE{1};                                              // The maximum default batch size for MNIST sample.
static const int32_t OUTPUT_SIZE{10};                                                // The output of the final MLP layer for MNIST sample.
static const nvinfer1::ActivationType MNIST_ACT{nvinfer1::ActivationType::kSIGMOID}; // The MNIST sample uses a sigmoid for activation.
static const char* INPUT_BLOB_NAME{"input"};                                         // The default input blob name.
static const char* OUTPUT_BLOB_NAME{"output"};                                       // the default output blob name.
static const char* DEFAULT_WEIGHT_FILE{"sampleMLP.wts2"};                            // The weight file produced from README.txt
samplesCommon::Args gArgs;

const std::string gSampleName = "TensorRT.sample_mlp";

/**
 * \class ShapedWeights
 * \brief A combination of Dims and Weights to provide shape to a weight struct.
 */
struct ShapedWeights
{
    nvinfer1::Dims shape;
    nvinfer1::Weights data;
};
typedef std::map<std::string, ShapedWeights> WeightMap_t;

// The split function takes string and based on a set of tokens produces a vector of tokens
// tokenized by the tokens. This is used to parse the shape field of the wts format.
static void split(std::vector<std::string>& split, std::string tokens, const std::string& input)
{
    split.clear();
    std::size_t begin = 0, size = input.size();
    while (begin != std::string::npos)
    {
        std::size_t found = input.find_first_of(tokens, begin);
        // Handle case of two or more delimiters in a row.
        if (found != begin)
            split.push_back(input.substr(begin, found - begin));
        begin = found + 1;
        // Handle case of no more tokens.
        if (found == std::string::npos)
            break;
        // Handle case of delimiter being last or first token.
        if (begin >= size)
            break;
    }
}

// Read a data blob from the input file.
void* loadShapeData(std::ifstream& input, size_t numElements)
{
    void* tmp = malloc(sizeof(float) * numElements);
    input.read(static_cast<char*>(tmp), numElements * sizeof(float));
    assert(input.peek() == '\n');
    // Consume the newline at the end of the data blob.
    input.get();
    return tmp;
}

nvinfer1::Dims loadShape(std::ifstream& input)
{
    // Initial format is "(A, B, C,...,Y [,])"
    nvinfer1::Dims shape{};
    std::string shapeStr;

    // Convert to "(A,B,C,...,Y[,])"
    do
    {
        std::string tmp;
        input >> tmp;
        shapeStr += tmp;
    } while (*shapeStr.rbegin() != ')');
    assert(input.peek() == ' ');

    // Consume the space between the shape and the data buffer.
    input.get();

    // Convert to "A,B,C,...,Y[,]"
    assert(*shapeStr.begin() == '(');
    shapeStr.erase(0, 1); //
    assert(*shapeStr.rbegin() == ')');
    shapeStr.pop_back();

    // Convert to "A,B,C,...,Y"
    if (*shapeStr.rbegin() == ',')
        shapeStr.pop_back(); // Remove the excess ',' character

    std::vector<std::string> shapeDim;
    split(shapeDim, ",", shapeStr);
    // Convert to {A, B, C,...,Y}
    assert(shapeDim.size() <= shape.MAX_DIMS);
    assert(shapeDim.size() > 0);
    assert(shape.nbDims == 0);
    std::for_each(shapeDim.begin(),
                  shapeDim.end(),
                  [&](std::string& val) {
                      shape.d[shape.nbDims++] = std::stoi(val);
                  });
    return shape;
}

// Our weight files are in a very simple space delimited format.
// type is the integer value of the DataType enum in NvInfer.h.
// <number of buffers>
// for each buffer: [name] [type] [size] <data x size in hex>
WeightMap_t loadWeights(const std::string file)
{
    WeightMap_t weightMap;
    std::ifstream input(file, std::ios_base::binary);
    assert(input.is_open() && "Unable to load weight file.");
    int32_t count;
    input >> count;
    assert(count > 0 && "Invalid weight map file.");
    while (count--)
    {
        ShapedWeights wt{};
        std::int32_t type;
        std::string name;
        input >> name >> std::dec >> type;
        wt.shape = loadShape(input);
        wt.data.type = static_cast<nvinfer1::DataType>(type);
        wt.data.count = std::accumulate(wt.shape.d, wt.shape.d + wt.shape.nbDims, 1, std::multiplies<int32_t>());
        assert(wt.data.type == nvinfer1::DataType::kFLOAT);
        wt.data.values = loadShapeData(input, wt.data.count);
        weightMap[name] = wt;
    }
    return weightMap;
}

// simple PGM (portable greyscale map) reader
void readPGMFile(const std::string& filename, uint8_t buffer[INPUT_H * INPUT_W])
{
    readPGMFile(locateFile(filename, gArgs.dataDirs), buffer, INPUT_H, INPUT_W);
}

// The addMLPLayer function is a simple helper function that creates the combination required for an
// MLP layer. By replacing the implementation of this sequence with various implementations, then
// then it can be shown how TensorRT optimizations those layer sequences.
nvinfer1::ILayer* addMLPLayer(nvinfer1::INetworkDefinition* network,
                              nvinfer1::ITensor& inputTensor,
                              int32_t hiddenSize,
                              nvinfer1::Weights wts,
                              nvinfer1::Weights bias,
                              nvinfer1::ActivationType actType,
                              int idx)
{
    std::string baseName("MLP Layer" + (idx == -1 ? "Output" : std::to_string(idx)));
    auto fc = network->addFullyConnected(inputTensor, hiddenSize, wts, bias);
    assert(fc != nullptr);
    std::string fcName = baseName + "FullyConnected";
    fc->setName(fcName.c_str());
    auto act = network->addActivation(*fc->getOutput(0), actType);
    assert(act != nullptr);
    std::string actName = baseName + "Activation";
    act->setName(actName.c_str());
    return act;
}

void transposeWeights(nvinfer1::Weights& wts, int hiddenSize)
{
    int d = 0;
    int dim0 = hiddenSize;       // 256 or 10
    int dim1 = wts.count / dim0; // 784 or 256
    uint32_t* trans_wts = new uint32_t[wts.count];
    for (int d0 = 0; d0 < dim0; ++d0)
    {
        for (int d1 = 0; d1 < dim1; ++d1)
        {
            trans_wts[d] = *((uint32_t*) wts.values + d1 * dim0 + d0);
            d++;
        }
    }

    for (int k = 0; k < wts.count; ++k)
    {
        *((uint32_t*) wts.values + k) = trans_wts[k];
    }
}

// Create the Engine using only the API and not any parser.
nvinfer1::ICudaEngine* fromAPIToModel(nvinfer1::IBuilder* builder)
{
    nvinfer1::DataType dt{nvinfer1::DataType::kFLOAT};
    WeightMap_t weightMap{loadWeights(locateFile(DEFAULT_WEIGHT_FILE, gArgs.dataDirs))};
    nvinfer1::INetworkDefinition* network = builder->createNetwork();

    // FC layers must still have 3 dimensions, so we create a {C, 1, 1,} matrix.
    // Currently the mnist example is only trained in FP32 mode.
    auto input = network->addInput(INPUT_BLOB_NAME, dt, nvinfer1::Dims3{(INPUT_H * INPUT_W), 1, 1});
    assert(input != nullptr);

    for (int i = 0; i < HIDDEN_COUNT; ++i)
    {
        std::stringstream weightStr, biasStr;
        weightStr << "hiddenWeights" << i;
        biasStr << "hiddenBias" << i;
        // Transpose hidden layer weights
        transposeWeights(weightMap[weightStr.str()].data, HIDDEN_SIZE);
        auto mlpLayer = addMLPLayer(network, *input, HIDDEN_SIZE, weightMap[weightStr.str()].data, weightMap[biasStr.str()].data, MNIST_ACT, i);
        input = mlpLayer->getOutput(0);
    }
    // Transpose output layer weights
    transposeWeights(weightMap["outputWeights"].data, OUTPUT_SIZE);

    auto finalLayer = addMLPLayer(network, *input, OUTPUT_SIZE, weightMap["outputWeights"].data, weightMap["outputBias"].data, MNIST_ACT, -1);
    assert(finalLayer != nullptr);
    // Run topK to get the final result
    auto topK = network->addTopK(*finalLayer->getOutput(0), nvinfer1::TopKOperation::kMAX, 1, 0x1);
    assert(topK != nullptr);
    topK->setName("OutputTopK");
    topK->getOutput(1)->setName(OUTPUT_BLOB_NAME);
    network->markOutput(*topK->getOutput(1));
    topK->getOutput(1)->setType(nvinfer1::DataType::kINT32);

    // Build the engine
    builder->setMaxBatchSize(MAX_BATCH_SIZE);
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setFp16Mode(gArgs.runInFp16);
    builder->setInt8Mode(gArgs.runInInt8);
    if (gArgs.runInInt8)
    {
        samplesCommon::setAllTensorScales(network, 64.0f, 64.0f);
    }
    
    samplesCommon::enableDLA(builder, gArgs.useDLACore);

    auto engine = builder->buildCudaEngine(*network);
    // we don't need the network any more
    network->destroy();

    // Once we have built the cuda engine, we can release all of our held memory.
    for (auto& mem : weightMap)
        free(const_cast<void*>(mem.second.data.values));
    return engine;
}

void APIToModel(nvinfer1::IHostMemory** modelStream)
{
    // create the builder
    nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger.getTRTLogger());
    assert(builder != nullptr);

    // create the model to populate the network, then set the outputs and create an engine
    nvinfer1::ICudaEngine* engine = fromAPIToModel(builder);

    assert(engine != nullptr);

    // GIE-3533
    // serialize the engine, then close everything down
    (*modelStream) = engine->serialize();
    engine->destroy();
    builder->destroy();
}

void doInference(nvinfer1::IExecutionContext& context, uint8_t* inputPtr, uint8_t* outputPtr)
{
    float* input = reinterpret_cast<float*>(inputPtr);
    int32_t* output = reinterpret_cast<int32_t*>(outputPtr);
    const nvinfer1::ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    assert(engine.getNbBindings() == 2);
    void* buffers[2];

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    CHECK(cudaMalloc(&buffers[inputIndex], MAX_BATCH_SIZE * (INPUT_H * INPUT_W) * 4));
    CHECK(cudaMalloc(&buffers[outputIndex], MAX_BATCH_SIZE * 4));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    // DMA the input to the GPU,  execute the batch asynchronously, and DMA it back:
    CHECK(cudaMemcpyAsync(buffers[inputIndex], input, MAX_BATCH_SIZE * (INPUT_H * INPUT_W) * 4, cudaMemcpyHostToDevice, stream));
    context.enqueue(MAX_BATCH_SIZE, buffers, stream, nullptr);
    CHECK(cudaMemcpyAsync(output, buffers[outputIndex], MAX_BATCH_SIZE * 4, cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);

    // release the stream and the buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
}

//!
//! \brief This function prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout << "Usage: ./sample_mlp [-h or --help] [-d or --datadir=<path to data directory>] [--useDLACore=<int>]\n";
    std::cout << "--help          Display help information\n";
    std::cout << "--datadir       Specify path to a data directory, overriding the default. This option can be used multiple times to add multiple directories. If no data directories are given, the default is to use (data/samples/mnist/, data/mnist/, data/samples/mlp/, data/mlp/)" << std::endl;
    std::cout << "--useDLACore=N  Specify a DLA engine for layers that support DLA. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "--int8          Run in Int8 mode.\n";
    std::cout << "--fp16          Run in FP16 mode." << std::endl;
}


int main(int argc, char* argv[])
{
    // create a model using the API directly and serialize it to a stream.
    nvinfer1::IHostMemory* modelStream{nullptr};
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
        gArgs.dataDirs = std::vector<std::string>{"data/samples/mnist/", "data/mnist/", "data/samples/mlp/", "data/mlp/"};
    }

    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    // Temporarily disable serialization path while debugging the layer.
    APIToModel(&modelStream);
    if (modelStream == nullptr)
    {
        gLogError << "Unable to create model." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(gLogger);
    if (runtime == nullptr)
    {
        gLogError << "Unable to create runtime." << std::endl;
        return gLogger.reportFail(sampleTest);
    }
    if (gArgs.useDLACore >= 0)
    {
        runtime->setDLACore(gArgs.useDLACore);
    }
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(modelStream->data(), modelStream->size(), nullptr);
    if (engine == nullptr)
    {
        gLogError << "Unable to build engine." << std::endl;
        return gLogger.reportFail(sampleTest);
    }
    modelStream->destroy();
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    if (context == nullptr)
    {
        gLogError << "Unable to create context." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    srand(unsigned(time(nullptr)));
    bool pass{true};
    int num = rand() % 10;
    // Just for simplicity, allocations for memory use float,
    // even for fp16 data type.
    uint8_t* input = new uint8_t[MAX_BATCH_SIZE * (INPUT_H * INPUT_W) * sizeof(float)];
    uint8_t* output = new uint8_t[MAX_BATCH_SIZE * sizeof(float)];
    if (input == nullptr || output == nullptr)
    {
        gLogError << "Host side memory allocation error." << std::endl;
        return gLogger.reportFail(sampleTest);
    }

    // read a random digit file from the data directory for use as input.
    auto fileData = new uint8_t[(INPUT_H * INPUT_W)];
    readPGMFile(std::to_string(num) + ".pgm", fileData);

    // print the ascii representation of the file that was loaded.
    gLogInfo << "Input:\n";
    for (int i = 0; i < (INPUT_H * INPUT_W); i++)
        gLogInfo << (" .:-=+*#%@"[fileData[i] / 26]) << (((i + 1) % INPUT_W) ? "" : "\n");
    gLogInfo << std::endl;

    // Normalize the data the same way TensorFlow does.
    for (int i = 0; i < (INPUT_H * INPUT_W); i++)
        reinterpret_cast<float*>(input)[i] = 1.0 - float(fileData[i]) / 255.0f;

    delete[] fileData;

    doInference(*context, input, output);

    int idx{*reinterpret_cast<int*>(output)};
    pass = (idx == num);
    if (pass)
        gLogInfo << "Algorithm chose " << idx << std::endl;
    else
        gLogInfo << "Algorithm chose " << idx << " but expected " << num << "." << std::endl;

    delete[] input;
    delete[] output;

    // destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();

    return gLogger.reportTest(sampleTest, pass);
}
