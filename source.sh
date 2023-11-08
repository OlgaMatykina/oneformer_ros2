source /opt/ros/foxy/setup.bash
cd ~/colcon_ws
colcon build --packages-select semseg_ros2 --symlink-install
source install/setup.bash 
ros2 launch semseg_ros2 oneformer_launch.py

# sudo apt-get install ros-foxy-image-view
cd ~/colcon_ws
source /opt/ros/foxy/setup.bash
ros2 run image_view image_saver --ros-args -r image:=/sensum/left/segmentation_color #/kitti/camera_color_left/segmentation_color 
-p filename_format:=image.jpg

source /opt/ros/noetic/setup.bash
source /opt/ros/foxy/setup.bash
ros2 bag play -r 0.07 -s rosbag_v2 camera_2023-07-26-09-55-05_2.bag