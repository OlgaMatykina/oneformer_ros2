#!/bin/bash

export CUDA_HOME="/usr/local/cuda-11.3"
cd ./src/semseg/oneformer/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
