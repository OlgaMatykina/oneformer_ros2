import launch
import launch.actions
import launch.substitutions
import launch_ros.actions


def generate_launch_description():
    return launch.LaunchDescription([
        # Параметры модели
        launch.actions.DeclareLaunchArgument(
            'cfg',
            default_value='/home/docker_oneformer_ros2/colcon_ws/src/semseg/configs/config_10cats.yaml'
        ),
        launch.actions.DeclareLaunchArgument(
            'cat_num',
            default_value='10'
        ),
        # launch.actions.DeclareLaunchArgument(
        #     'treshold',
        #     default_value='0.5'
        # ),

        # Настройка топиков
        launch.actions.DeclareLaunchArgument(
            'camera_ns',
            # default_value='/kitti/camera_color_left/'
            default_value='/sensum/left/'
        ),
        launch.actions.DeclareLaunchArgument(
            'image_topic',
            default_value='image_raw'
            # default_value='image_rect_color'
        ),
        launch.actions.DeclareLaunchArgument(
            'segmentation_topic',
            default_value='segmentation'
        ),
        launch.actions.DeclareLaunchArgument(
            'segmentation_color_topic',
            default_value='segmentation_color'
        ),

        # Nodes
        launch_ros.actions.Node(
            package='semseg_ros2',
            namespace=launch.substitutions.LaunchConfiguration('camera_ns'),
            executable='oneformer_node',
            name='oneformer_node',
            remappings=[
                ('image', launch.substitutions.LaunchConfiguration('image_topic')),
                ('segmentation', launch.substitutions.LaunchConfiguration('segmentation_topic')),
            ],
            parameters=[
                {
                    'cfg': launch.substitutions.LaunchConfiguration('cfg'),
                    'cat_num': launch.substitutions.LaunchConfiguration('cat_num'),
                    # 'treshold': launch.substitutions.LaunchConfiguration('treshold')
                }
            ],
            output="screen"
        ),

        launch_ros.actions.Node(
            package='semseg_ros2',
            namespace=launch.substitutions.LaunchConfiguration('camera_ns'),
            executable='visualizer_node',
            name='visualizer_node',
            remappings=[
                ('image', launch.substitutions.LaunchConfiguration('image_topic')),
                ('segmentation', launch.substitutions.LaunchConfiguration('segmentation_topic')),
                ('segmentation_color', launch.substitutions.LaunchConfiguration('segmentation_color_topic'))
            ],
            parameters=[
                {
                    'cat_num': launch.substitutions.LaunchConfiguration('cat_num'),
                    # 'treshold': launch.substitutions.LaunchConfiguration('treshold')
                }
            ],
            output="screen"
        )
    ])
