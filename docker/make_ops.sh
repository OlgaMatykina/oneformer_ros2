#!/bin/bash

export CUDA_HOME="/usr/local/cuda-11.3"
cd OneFormer/oneformer/modeling/pixel_decoder/ops
sh make.sh
cd ../../../..
