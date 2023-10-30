#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

cd "$(dirname "$0")"
root_dir=$PWD 

to_archive () {
tar --remove-files -czf $1.tar.gz $1 
}

if [[ -d "$root_dir/context" ]]
then
    echo "${orange}${root_dir}/context${reset_color} exists on your filesystem. Delete it to update docker context folder"
else
    mkdir $root_dir/context
    cd $root_dir/context
    echo "Download docker context to ${orange}${PWD}/context${reset_color}"

    tput setaf 2

    echo "Cloning"
    # Place your additional dependencies for image build which place in private repos

    # Example:
    # git clone https://gitlab.com/sdbcs-nio3/alg/slam/3rd_party_libs/pcl.git --branch pcl-1.10.1/fix_mesh_concatenation
    # to_archive pcl
    # git clone https://github.com/ros-perception/perception_pcl.git --branch melodic-devel
    # to_archive perception_pcl

    echo "Downloading"
    # Place your additional dependencies for image build which can be downloaded

    # Example:
    # wget -O ceres-solver-1.14.0.tar.gz ceres-solver.org/ceres-solver-1.14.0.tar.gz
    # wget -O gtsam-4.0.3.zip https://github.com/borglab/gtsam/archive/4.0.3.zip

    tput sgr0
fi
