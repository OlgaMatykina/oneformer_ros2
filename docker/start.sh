#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

export ARCH=`uname -m`

cd "$(dirname "$0")"
root_dir=$PWD 
cd $root_dir

echo "Running on ${orange}${ARCH}${reset_color}"

if [ "$ARCH" == "x86_64" ] 
then
    ARGS="--ipc host --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
elif [ "$ARCH" == "aarch64" ] 
then
    ARGS="--runtime nvidia"
else
    echo "Arch ${ARCH} not supported"
    exit
fi

xhost +
docker run -it -d --rm \
        $ARGS \
        --env="DISPLAY=$DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name oneformer_ros2 \
        --net "host" \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v /home/izmesteva_u/Desktop/oneformer_ros2/colcon_ws:/home/docker_oneformer_ros2/colcon_ws:rw \
        ${ARCH}foxy/semseg:latest
xhost -
