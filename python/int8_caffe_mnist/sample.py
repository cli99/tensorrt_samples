#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

import numpy as np
import random

# For our custom calibrator
import calibrator

# For ../common.py
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], os.path.pardir))
import common

TRT_LOGGER = trt.Logger()

class ModelData(object):
    DEPLOY_PATH = "deploy.prototxt"
    MODEL_PATH = "mnist_lenet.caffemodel"
    OUTPUT_NAME = "prob"
    # The original model is a float32 one.
    DTYPE = trt.float32

# This function builds an engine from a Caffe model.
def build_int8_engine(deploy_file, model_file, calib):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        # We set the builder batch size to be the same as the calibrator's, as we use the same batches
        # during inference. Note that this is not required in general, and inference batch size is
        # independent of calibration batch size.
        builder.max_batch_size = calib.get_batch_size()
        builder.max_workspace_size = common.GiB(1)
        builder.int8_mode = True
        builder.int8_calibrator = calib
        # Parse Caffe model
        model_tensors = parser.parse(deploy=deploy_file, model=model_file, network=network, dtype=ModelData.DTYPE)
        network.mark_output(model_tensors.find(ModelData.OUTPUT_NAME))
        # Build engine and do int8 calibration.
        return builder.build_cuda_engine(network)

# Loads a random batch from the supplied calibrator.
def load_random_batch(calib):
    # Load a random batch.
    batch = random.choice(calib.batch_files)
    _, data, labels = calib.read_batch_file(batch)
    data = np.fromstring(data, dtype=np.float32)
    labels = np.fromstring(labels, dtype=np.float32)
    return data, labels

# Note that we don't expect the accuracy to be 100%, but it should
# be close to the fp32 model (which is in the 98-99% range).
def validate_output(output, labels):
    preds = np.argmax(output, axis=1)
    print("Expected Predictons:\n" + str(labels))
    print("Actual Predictons:\n" + str(preds))
    check = np.equal(preds, labels)
    accuracy = np.sum(check) / float(check.size) * 100
    print("Accuracy: " + str(accuracy) + "%")
    # If a prediction was incorrect print out an array of booleans indicating accuracy.
    if accuracy != 100:
        print("One or more predictions was incorrect:\n" + str(check))

def main():
    data_path, data_files = common.find_sample_data(description="Runs a Caffe MNIST network in Int8 mode", subfolder="mnist", find_files=["batches", ModelData.DEPLOY_PATH, ModelData.MODEL_PATH])
    [batch_data_dir, deploy_file, model_file] = data_files

    # Now we create a calibrator and give it the location of our calibration data.
    # We also allow it to cache calibration data for faster engine building.
    calibration_cache = "mnist_calibration.cache"
    calib = calibrator.MNISTEntropyCalibrator(batch_data_dir, cache_file=calibration_cache)
    # We will use the calibrator batch size across the board.
    # This is not a requirement, but in this case it is convenient.
    batch_size = calib.get_batch_size()

    with build_int8_engine(deploy_file, model_file, calib) as engine, engine.create_execution_context() as context:
        # Allocate engine buffers.
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)

        # Do inference for the whole batch. We have to specify batch size here, as the common.do_inference uses a default
        inputs[0].host, labels = load_random_batch(calib)
        [output] = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, batch_size=batch_size)
        # Next we need to reshape the output to Nx10 (10 probabilities, one per digit), where N is batch size.
        output = output.reshape(batch_size, 10)
        validate_output(output, labels)

if __name__ == '__main__':
    main()
