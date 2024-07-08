# FROM puzlcloud/pytorch:1.10.1-cuda11.3-cudnn8-jupyter-g1-1.1.0-python3.8
FROM nvcr.io/nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for convinient development inside container
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    git \
    tmux \
    curl \
    wget \
    build-essential \
    sudo \
    python3-pip
    # && rm -rf /var/lib/apt/lists/*


# Install PyTorch
RUN python3 -m pip install --upgrade pip
# RUN python3 -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
RUN python3 -m pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# Instal basic utilities

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    apt-get update && \
    apt-get install -y --no-install-recommends python3.8-dev git wget unzip bzip2 sudo build-essential ca-certificates && \
    apt-get install ffmpeg libsm6 libxext6  -y && \
     apt-get clean
    # && rm -rf /var/lib/apt/lists/*

# Install ROS2 Foxy
RUN apt-get update \
    && apt-get install -y curl gnupg2 lsb-release \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key  -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y \
        ros-foxy-ros-base \
        python3-colcon-common-extensions \
        ros-foxy-cv-bridge
    # && rm -rf /var/lib/apt/lists/*

# RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# RUN apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# RUN apt update
# RUN apt install -y \
#     ros-foxy-ros2bag \
#     ros-foxy-rosbag2 \
#     ros-foxy-ros1-bridge \
#     ros-foxy-rosbag2-bag-v2-plugins \
#     ros-foxy-rosbag2-converter-default-plugins


# RUN python3 -m pip install opencv-python ultralytics

# add user and his password
ARG USER=docker_oneformer_ros2
ARG UID=1000
ARG GID=1000
# default password
ARG PW=user

ENV PYTHONPATH=/home/${USER}/colcon_ws/src/semseg:${PYTHONPATH}

RUN useradd -m ${USER} --uid=${UID} && echo "${USER}:${PW}" | chpasswd && adduser ${USER} sudo
WORKDIR /home/${USER}
RUN mkdir -p colcon_ws/src && chown -R ${UID}:${GID} /home/${USER}
USER ${UID}:${GID}

RUN pip3 install -U opencv-python
RUN pip3 install -U pycocotools
# RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# RUN pip3 install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu113/torch1.10/index.html
# RUN python3 -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
RUN python3 -m pip install -U 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02'
# RUN pip3 install git+https://github.com/cocodataset/panopticapi.git
# RUN pip3 install git+https://github.com/mcordts/cityscapesScripts.git
# RUN git clone https://github.com/SHI-Labs/OneFormer.git 
# RUN git clone --branch master https://github.com/OlgaMatykina/oneformer_ros2
# COPY ./requirements.txt ./requirements_oneformer.txt
# RUN cd OneFormer && \
COPY ./requirements.txt ./requirements_oneformer.txt
RUN pip3 install -r ./requirements_oneformer.txt
#     pip3 install wandb
