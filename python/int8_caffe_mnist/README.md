# INT8 Calibration in Python

**Table Of Contents**
- [Description](#description)
- [How does this sample work?](#how-does-this-sample-work)
- [Prerequisites](#prerequisites)
- [Running the sample](#running-the-sample)
	* [Sample `--help` options](#sample---help-options)
- [Additional resources](#additional-resources)
- [License](#license)
- [Changelog](#changelog)
- [Known issues](#known-issues)

## Description

This sample, int8_caffe_mnist, demonstrates how to create an INT8 calibrator, build and calibrate an engine for INT8 mode, and finally run inference in INT8 mode.

## How does this sample work?

During calibration, the calibrator retrieves a total of 1003 batches, with 100 images each. We have simplified the process of reading and writing a calibration cache in Python, so that it is now easily possible to cache calibration data to speed up engine builds (see `calibrator.py` for implementation details).

During inference, the sample loads a random batch from the calibrator, then performs inference on the whole batch of 100 images.

## Prerequisites

1. Install the dependencies for Python.
	-   For Python 2 users, from the root directory, run:
		`python2 -m pip install -r requirements.txt`

	-   For Python 3 users, from the root directory, run:
		`python3 -m pip install -r requirements.txt`

## Running the sample

1.  Run the sample to create a TensorRT inference engine, perform IN8 calibration and run inference:
	`python3 sample.py [-d DATA_DIR]`

	to run the sample with Python 3.

	**Note:** If the TensorRT sample data is not installed in the default location, for example `/usr/src/tensorrt/data/`, the `data` directory must be specified. For example:
	`python sample.py -d /path/to/my/data/`.


2.  Verify that the sample ran successfully. If the sample runs successfully you should see a very high accuracy. For example:
	```
	Expected Predictions:
	[1. 6. 5. 0. 2. 8. 1. 5. 6. 2. 3. 0. 2. 2. 6. 4. 3. 5. 5. 1. 7. 2. 1. 6.
	9. 1. 9. 9. 5. 5. 1. 6. 2. 2. 8. 6. 7. 1. 4. 6. 0. 4. 0. 3. 3. 2. 2. 3.
	6. 8. 9. 8. 5. 3. 8. 5. 4. 5. 2. 0. 5. 6. 3. 2. 8. 3. 9. 9. 5. 7. 9. 4.
	6. 7. 1. 3. 7. 3. 6. 6. 0. 9. 0. 1. 9. 9. 2. 8. 8. 0. 1. 6. 9. 7. 5. 3.
	4. 7. 4. 9.]
	Actual Predictions:
	[1 6 5 0 2 8 1 5 6 2 3 0 2 2 6 4 3 5 5 1 7 2 1 6 9 1 9 9 5 5 1 6 2 2 8 6 7
	1 4 6 0 4 0 3 3 2 2 3 6 8 9 8 5 3 8 5 4 5 2 0 5 6 3 2 8 3 9 9 5 7 9 4 6 7
	1 3 7 3 6 6 0 9 0 1 9 4 2 8 8 0 1 6 9 7 5 3 4 7 4 9]
	Accuracy: 99.0%
	```

### Sample --help options

To see the full list of available options and their descriptions, use the `-h` or `--help` command line option. For example:
```
usage: sample.py [-h]

Description for this sample

optional arguments:
	-h, --help show this help message and exit
```

# Additional resources

The following resources provide a deeper understanding about the model used in this sample:

**Network**
- [MNIST network](http://yann.lecun.com/exdb/lenet/)

**Dataset**
- [MNIST dataset](http://yann.lecun.com/exdb/mnist/)

**Documentation**
- [Introduction to NVIDIA’s TensorRT Samples](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#samples)
- [Working with TensorRT Using the Python API](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#python_topics)
- [Enabling INT8 Inference Using Python](https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html#enable_int8_python)
- [NVIDIA’s TensorRT Documentation Library](https://docs.nvidia.com/deeplearning/sdk/tensorrt-archived/index.html)

# License

For terms and conditions for use, reproduction, and distribution, see the [TensorRT Software License Agreement](https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html) documentation.

# Changelog

March 2019
This `README.md` file was recreated, updated and reviewed.

# Known issues

There are no known issues in this sample.
