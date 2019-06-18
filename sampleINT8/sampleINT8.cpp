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

#include <cassert>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <sys/stat.h>
#include <cmath>
#include <time.h>
#include <cuda_runtime_api.h>
#include <unordered_map>
#include <algorithm>
#include <float.h>
#include <string.h>
#include <chrono>
#include <iterator>
#include <exception>

#include "NvInfer.h"
#include "NvCaffeParser.h"
#include "common.h"

#include "logger.h"
#include "BatchStream.h"
#include "LegacyCalibrator.h"
#include "EntropyCalibrator.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;

const std::string gSampleName = "TensorRT.sample_int8";

static int gUseDLACore = -1;

// stuff we know about the network and the caffe input/output blobs

const char* INPUT_BLOB_NAME = "data";
const char* OUTPUT_BLOB_NAME = "prob";
const char* gNetworkName{nullptr};

struct SkipThePrecision : public std::exception
{
    const char* what() const noexcept
    {
        return "Spcified precision is not natively support";
    }
};

std::string locateFile(const std::string& input)
{
    std::vector<std::string> dirs;
    dirs.push_back(std::string("data/") + gNetworkName + std::string("/"));
    dirs.push_back(std::string("int8/") + gNetworkName + std::string("/"));
    dirs.push_back(std::string("data/int8/") + gNetworkName + std::string("/"));
    dirs.push_back(std::string("data/int8_samples/") + gNetworkName + std::string("/"));
    return locateFile(input, dirs);
}

bool caffeToTRTModel(const std::string& deployFile,           // name for caffe prototxt
                     const std::string& modelFile,            // name for model
                     const std::vector<std::string>& outputs, // network outputs
                     int& maxBatchSize,                       // batch size - NB must be at least as large as the batch we want to run with)
                     DataType dataType,
                     IInt8Calibrator* calibrator,
                     nvinfer1::IHostMemory*& trtModelStream)
{
    // create the builder
    IBuilder* builder = createInferBuilder(gLogger.getTRTLogger());
    if (builder == nullptr)
    {
        return false;
    }

    // parse the caffe model to populate the network, then set the outputs
    INetworkDefinition* network = builder->createNetwork();
    ICaffeParser* parser = createCaffeParser();

    if ((dataType == DataType::kINT8 && !builder->platformHasFastInt8()) || (dataType == DataType::kHALF && !builder->platformHasFastFp16()))
    {
        parser->destroy();
        network->destroy();
        builder->destroy();
        throw SkipThePrecision();
        return false;
    }
    std::string dFile{locateFile(deployFile)};
    if (dFile.empty())
    {
        return false;
    }
    std::string mFile{locateFile(modelFile)};
    if (mFile.empty())
    {
        return false;
    }
    const IBlobNameToTensor* blobNameToTensor = parser->parse(dFile.c_str(),
                                                              mFile.c_str(),
                                                              *network,
                                                              dataType == DataType::kINT8 ? DataType::kFLOAT : dataType);

    // specify which tensors are outputs
    for (auto& s : outputs)
    {
        network->markOutput(*blobNameToTensor->find(s.c_str()));
    }

    // Build the engine
    builder->setMaxWorkspaceSize(1 << 30);
    builder->setAverageFindIterations(1);
    builder->setMinFindIterations(1);
    builder->setDebugSync(true);
    builder->setInt8Mode(dataType == DataType::kINT8);
    builder->setFp16Mode(dataType == DataType::kHALF);
    builder->setInt8Calibrator(calibrator);
    if (gUseDLACore >= 0)
    {
        samplesCommon::enableDLA(builder, gUseDLACore);
        if (maxBatchSize > builder->getMaxDLABatchSize())
        {
            gLogError << "Requested batch size " << maxBatchSize << " is greater than the max DLA batch size of "
                      << builder->getMaxDLABatchSize() << ". Reducing batch size accordingly." << std::endl;
            maxBatchSize = builder->getMaxDLABatchSize();
        }
    }
    builder->setMaxBatchSize(maxBatchSize);
    ICudaEngine* engine = builder->buildCudaEngine(*network);
    if (engine == nullptr)
    {
        gLogError << "Unable to build engine." << std::endl;
    }

    // we don't need the network any more, and we can destroy the parser
    network->destroy();
    parser->destroy();

    if (engine)
    {
        // serialize the engine, then close everything down
        trtModelStream = engine->serialize();
        engine->destroy();
    }
    builder->destroy();
    return true;
}

float doInference(IExecutionContext& context, float* input, float* output, int batchSize)
{
    const ICudaEngine& engine = context.getEngine();
    // input and output buffer pointers that we pass to the engine - the engine requires exactly IEngine::getNbBindings(),
    // of these, but in this case we know that there is exactly one input and one output.
    void* buffers[2];
    float ms{0.0f};

    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // note that indices are guaranteed to be less than IEngine::getNbBindings()
    int inputIndex = engine.getBindingIndex(INPUT_BLOB_NAME),
        outputIndex = engine.getBindingIndex(OUTPUT_BLOB_NAME);

    // create GPU buffers and a stream
    Dims3 inputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(INPUT_BLOB_NAME)));
    Dims3 outputDims = static_cast<Dims3&&>(context.getEngine().getBindingDimensions(context.getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));

    size_t inputSize = batchSize * inputDims.d[0] * inputDims.d[1] * inputDims.d[2] * sizeof(float), outputSize = batchSize * outputDims.d[0] * outputDims.d[1] * outputDims.d[2] * sizeof(float);
    CHECK(cudaMalloc(&buffers[inputIndex], inputSize));
    CHECK(cudaMalloc(&buffers[outputIndex], outputSize));

    CHECK(cudaMemcpy(buffers[inputIndex], input, inputSize, cudaMemcpyHostToDevice));

    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    cudaEvent_t start, end;
    CHECK(cudaEventCreateWithFlags(&start, cudaEventBlockingSync));
    CHECK(cudaEventCreateWithFlags(&end, cudaEventBlockingSync));
    cudaEventRecord(start, stream);
    context.enqueue(batchSize, buffers, stream, nullptr);
    cudaEventRecord(end, stream);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&ms, start, end);
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    CHECK(cudaMemcpy(output, buffers[outputIndex], outputSize, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    CHECK(cudaStreamDestroy(stream));
    return ms;
}

int calculateScore(float* batchProb, float* labels, int batchSize, int outputSize, int threshold)
{
    int success = 0;
    for (int i = 0; i < batchSize; i++)
    {
        float *prob = batchProb + outputSize * i, correct = prob[(int) labels[i]];

        int better = 0;
        for (int j = 0; j < outputSize; j++)
        {
            if (prob[j] >= correct)
            {
                better++;
            }
        }
        if (better <= threshold)
        {
            success++;
        }
    }
    return success;
}

std::pair<float, float> scoreModel(int batchSize, int firstBatch, int nbScoreBatches, DataType datatype, IInt8Calibrator* calibrator, bool quiet = false)
{
    IHostMemory* trtModelStream{nullptr};
    bool valid = false;
    if (gNetworkName == std::string("mnist"))
    {
        valid = caffeToTRTModel("deploy.prototxt", "mnist_lenet.caffemodel", std::vector<std::string>{OUTPUT_BLOB_NAME}, batchSize, datatype, calibrator, trtModelStream);
    }
    else
    {
        valid = caffeToTRTModel("deploy.prototxt", std::string(gNetworkName) + ".caffemodel", std::vector<std::string>{OUTPUT_BLOB_NAME}, batchSize, datatype, calibrator, trtModelStream);
    }

    if (!valid)
    {
        gLogError << "Engine could not be created at this precision" << std::endl;
        return std::pair<float, float>(0, 0);
    }
    if (trtModelStream == nullptr)
    {
        gLogError << "Network deserialization error." << std::endl;
        return std::make_pair(0.0f, 0.0f);
    }

    // Create engine and deserialize model.
    IRuntime* infer = createInferRuntime(gLogger.getTRTLogger());
    if (infer == nullptr)
    {
        gLogError << "Unable to create inference runtime." << std::endl;
        return std::make_pair(0.0f, 0.0f);
    }
    if (gUseDLACore >= 0)
    {
        infer->setDLACore(gUseDLACore);
    }
    ICudaEngine* engine = infer->deserializeCudaEngine(trtModelStream->data(), trtModelStream->size(), nullptr);
    if (engine == nullptr)
    {
        gLogError << "Unable to deserializeCudaEngine." << std::endl;
        return std::make_pair(0.0f, 0.0f);
    }
    trtModelStream->destroy();
    IExecutionContext* context = engine->createExecutionContext();
    if (context == nullptr)
    {
        gLogError << "Unable to create executionContext()." << std::endl;
        return std::make_pair(0.0f, 0.0f);
    }

    BatchStream stream(batchSize, nbScoreBatches);
    stream.skip(firstBatch);

    Dims3 outputDims = static_cast<Dims3&&>(context->getEngine().getBindingDimensions(context->getEngine().getBindingIndex(OUTPUT_BLOB_NAME)));
    int outputSize = outputDims.d[0] * outputDims.d[1] * outputDims.d[2];
    int top1{0}, top5{0};
    float totalTime{0.0f};
    std::vector<float> prob(batchSize * outputSize, 0);

    while (stream.next())
    {
        totalTime += doInference(*context, stream.getBatch(), &prob[0], batchSize);

        top1 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 1);
        top5 += calculateScore(&prob[0], stream.getLabels(), batchSize, outputSize, 5);

        if (stream.getBatchesRead() % 100 == 0)
        {
            if (quiet)
            {
                gLogVerbose << "Processing next set of max 100 batches" << std::endl;
            }
            else
            {
                gLogInfo << "Processing next set of max 100 batches" << std::endl;
            }
        }
    }
    int imagesRead = stream.getBatchesRead() * batchSize;
    float t1 = float(top1) / float(imagesRead), t5 = float(top5) / float(imagesRead);

    if (quiet)
    {
        gLogVerbose << "Top1: " << t1 << ", Top5: " << t5 << std::endl;
        gLogVerbose << "Processing " << imagesRead << " images averaged " << totalTime / imagesRead << " ms/image and " << totalTime / stream.getBatchesRead() << " ms/batch." << std::endl;
    }
    else
    {
        gLogInfo << "Top1: " << t1 << ", Top5: " << t5 << std::endl;
        gLogInfo << "Processing " << imagesRead << " images averaged " << totalTime / imagesRead << " ms/image and " << totalTime / stream.getBatchesRead() << " ms/batch." << std::endl;
    }

    context->destroy();
    engine->destroy();
    infer->destroy();
    return std::make_pair(t1, t5);
}

std::pair<float, float> scoreModel(int batchSize, int firstScoreBatch, int nbScoreBatches, CalibrationAlgoType calibrationAlgo, bool search)
{
    std::unique_ptr<IInt8Calibrator> calibrator;
    BatchStream calibrationStream(CAL_BATCH_SIZE, NB_CAL_BATCHES);
    if (calibrationAlgo == CalibrationAlgoType::kENTROPY_CALIBRATION)
    {
        calibrator.reset(new Int8EntropyCalibrator(calibrationStream, FIRST_CAL_BATCH, gNetworkName, INPUT_BLOB_NAME));
    }
    else if (calibrationAlgo == CalibrationAlgoType::kLEGACY_CALIBRATION)
    {
        std::pair<double, double> parameters = getQuantileAndCutoff(gNetworkName, search);
        calibrator.reset(new Int8LegacyCalibrator(calibrationStream, FIRST_CAL_BATCH, parameters.first, parameters.second));
    }
    else
    {
        calibrator.reset(new Int8EntropyCalibrator2(calibrationStream, FIRST_CAL_BATCH, gNetworkName, INPUT_BLOB_NAME));
    }
    return scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kINT8, calibrator.get());
}

static void printUsage()
{
    std::cout << "Usage: ./sample_int8 <network name> <optional params>" << std::endl;
    std::cout << std::endl;
    std::cout << "Optional params" << std::endl;
    std::cout << "  --batch=N            Set batch size (default = 100)." << std::endl;
    std::cout << "  --start=N            Set the first batch to be scored (default = 100). All batches before this batch will be used for calibration." << std::endl;
    std::cout << "  --score=N            Set the number of batches to be scored (default = 400)." << std::endl;
    std::cout << "  --search             Search for best calibration. Can only be used with legacy calibration algorithm." << std::endl;
    std::cout << "  --legacy             Use legacy calibration algorithm." << std::endl;
    std::cout << "  --useLegacyEntropy   Use legacy Entropy calibration algorithm." << std::endl;
    std::cout << "  --useDLACore=N       Enable execution on DLA for all layers that support dla. Value can range from 0 to n-1, where n is the number of DLA engines on the platform." << std::endl;
    std::cout << "  -h --help            Print this help menu." << std::endl;
}

int main(int argc, char** argv)
{
    bool pass{true};
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));

    gLogger.reportTestStart(sampleTest);

    if (argc < 2 || !strncmp(argv[1], "help", 4) || !strncmp(argv[1], "--help", 6) || !strncmp(argv[1], "-h", 2))
    {
        printUsage();
        return gLogger.reportTest(sampleTest, pass);
    }

    gNetworkName = argv[1];

    // by default we score over 40000 images starting at 10000, so we don't score those used to search calibration
    int batchSize = 100, firstScoreBatch = 100, nbScoreBatches = 400;
    bool search = false, batchSizeProvided = false;
    CalibrationAlgoType calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION_2;

    for (int i = 2; i < argc; i++)
    {
        if (!strncmp(argv[i], "--batch=", 8))
        {
            batchSize = atoi(argv[i] + 8);
            batchSizeProvided = true;
        }
        else if (!strncmp(argv[i], "--start=", 8))
        {
            firstScoreBatch = atoi(argv[i] + 8);
        }
        else if (!strncmp(argv[i], "--score=", 8))
        {
            nbScoreBatches = atoi(argv[i] + 8);
        }
        else if (!strncmp(argv[i], "--search", 8))
        {
            search = true;
        }
        else if (!strncmp(argv[i], "--help", 6) || !strncmp(argv[i], "-h", 2))
        {
            printUsage();
            return gLogger.reportTest(sampleTest, pass);
        }
        else if (!strncmp(argv[i], "--legacy", 8))
        {
            calibrationAlgo = CalibrationAlgoType::kLEGACY_CALIBRATION;
        }
        else if (!strncmp(argv[i], "--useLegacyEntropy", 18))
        {
            calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION;
        }
        else if (!strncmp(argv[i], "--useDLACore=", 13))
        {
            gUseDLACore = stoi(argv[i] + 13);
        }
        else
        {
            gLogError << "Unrecognized argument " << argv[i] << std::endl;
            pass = false;
        }
    }

    if (batchSize > 128)
    {
        gLogError << "Please provide batch size <= 128" << std::endl;
        pass = false;
    }

    if ((firstScoreBatch + nbScoreBatches) * batchSize > 500000)
    {
        gLogError << "Only 50000 images available" << std::endl;
        pass = false;
    }

    //if the cli is not set properly, return early as failure
    if (pass == false)
    {
   	 return gLogger.reportTest(sampleTest, pass);
    }
    gLogVerbose.precision(6);
    gLogInfo.precision(6);
    gLogError.precision(6);
    int dla{gUseDLACore};

    // Set gUseDLACore to -1 here since FP16 mode is not enabled.
    if (gUseDLACore >= 0)
    {
        if (!batchSizeProvided)
        {
            //  Set the default batch size to 16 for DLA..
            batchSize = 16;
        }
        gLogInfo << "DLA requested. Disabling for FP32 run since its not supported." << std::endl;
        gUseDLACore = -1;
    }

    std::pair<float, float> fp32Score, fp16Score, int8Score;
    gLogInfo << "FP32 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
    fp32Score = scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kFLOAT, nullptr);

    // Set gUseDLACore correctly to enable DLA if requested.
    gUseDLACore = dla;
    bool fp16Skipped = false;
    gLogInfo << "FP16 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
    try
    {
        fp16Score = scoreModel(batchSize, firstScoreBatch, nbScoreBatches, DataType::kHALF, nullptr);
    }
    catch (SkipThePrecision e)
    {
        gLogInfo << e.what() << std::endl;
        fp16Skipped = true;
    }

    // Use gUseDLACore correctly with INT8 mode if DLA is requested.
    if (gUseDLACore >= 0)
    {
        if (calibrationAlgo != CalibrationAlgoType::kENTROPY_CALIBRATION_2)
        {
            gLogInfo << "\nUser requested Legacy Calibration with DLA. DLA only supports kENTROPY_CALIBRATOR_2.";
        }
        gLogInfo << "\nDLA requested. Setting Calibrator to use kENTROPY_CALIBRATOR_2." << std::endl;
        calibrationAlgo = CalibrationAlgoType::kENTROPY_CALIBRATION_2;
    }
    gLogInfo << "\nINT8 run:" << nbScoreBatches << " batches of size " << batchSize << " starting at " << firstScoreBatch << std::endl;
    bool int8Skipped = false;
    try
    {
        int8Score = scoreModel(batchSize, firstScoreBatch, nbScoreBatches, calibrationAlgo, search);
    }
    catch (SkipThePrecision e)
    {
        gLogInfo << e.what() << std::endl;
        int8Skipped = true;
    }

    auto isApproximatelyEqual = [](float a, float b, double tolerance)
    {
        return (std::abs(a - b) <= tolerance);
    };
    double fp16tolerance{0.5}, int8tolerance{1.0};

    if (fp32Score.first == 0 && fp32Score.second == 0)
    {
        gLogError << "Unable to build engine for FP32." << std::endl;
        pass = false;
    }
    if (!fp16Skipped && !isApproximatelyEqual(fp32Score.first, fp16Score.first, fp16tolerance))
    {
        gLogError << "FP32(" << fp32Score.first << ") and FP16(" << fp16Score.first << ") Top1 accuracy differ by more than " << fp16tolerance << "." << std::endl;
        pass = false;
    }
    if (!int8Skipped && !isApproximatelyEqual(fp32Score.first, int8Score.first, int8tolerance))
    {
        gLogError << "FP32(" << fp32Score.first << ") and Int8(" << int8Score.first << ") Top1 accuracy differ by more than " << int8tolerance << "." << std::endl;
        pass = false;
    }
    if (!fp16Skipped && !isApproximatelyEqual(fp32Score.second, fp16Score.second, fp16tolerance))
    {
        gLogError << "FP32(" << fp32Score.second << ") and FP16(" << fp16Score.second << ") Top5 accuracy differ by more than " << fp16tolerance << "." << std::endl;
        pass = false;
    }
    if (!int8Skipped && !isApproximatelyEqual(fp32Score.second, int8Score.second, int8tolerance))
    {
        gLogError << "FP32(" << fp32Score.second << ") and INT8(" << int8Score.second << ") Top5 accuracy differ by more than " << int8tolerance << "." << std::endl;
        pass = false;
    }

    shutdownProtobufLibrary();

    return gLogger.reportTest(sampleTest, pass);
}
