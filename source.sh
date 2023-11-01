source /opt/ros/foxy/setup.bash
colcon build --packages-select semseg_ros2 --symlink-install
source install/setup.bash 
ros2 launch semseg_ros2 oneformer_launch.py

sudo apt-get install ros-foxy-image-view
ros2 run image_view image_saver --ros-args -r image:=/sensum/left/segmentation_color #/kitti/camera_color_left/segmentation_color 
-p filename_format:=image.jpg
