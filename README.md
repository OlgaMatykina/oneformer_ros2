# oneformer_ros2

Создание образа из докер файла:
```
./build.sh
```

Запуск контейнера (предварительно заменить volumes):
```
./start.sh
```

Подключение к контейнеру:
```
./into.sh
```
Для того, чтобы настроить CUDA Kernel для MSDeformAttn выполнить следующие команды в контейнере:
```
cd ~/colcon_ws
./make_ops.sh
pip install torchinfo
```

## Работа с готовым пакетом

В настоящем репозитории представлен готовый к использованию ROS2 пакет семантической сегментации OneFormer.

К данному моменту предполагается, что собран образ, запущен контейнер и выполнен вход в него.

Сначала необходимо собрать пакет:

```
cd ~/colcon_ws
source /opt/ros/foxy/setup.bash
colcon build --packages-select semseg_ros2 --symlink-install
source install/setup.bash 
```
После этого необходимо скачать веса по ссылке и поместить файл в папку 
```
~/oneformer_ros2/colcon_ws/src/semseg/weights
```
Затем нужно открыть конфигурационный файл, который расположен
```
~/oneformer_ros2/colcon_ws/src/semseg/weights/valid/swin/oneformer_swin_large_sem_seg_bs4_640k.yaml
```
и изменить название файла с весами (раскомментировать одну из строк):
```
  # WEIGHTS: /home/docker_oneformer_ros2/colcon_ws/src/semseg/weights/train422_steps120k.pth
  # WEIGHTS: /home/docker_oneformer_ros2/colcon_ws/src/semseg/weights/train1723_steps210k.pth
```

Затем запустить launch, который автоматически запустит необходимые компоненты, передав в качестве аргумента image_topic:
```
ros2 launch semseg_ros2 oneformer_launch.py image_topic:=image_rect_color
```
Для тестирования работы узла нужно поместить ROS-bag в папку ~/oneformer_ros2/colcon_ws.
Для запуска проигрывания нужно сначала активировать окружение ROS1, затем ROS2:
```
cd ~/colcon_ws
source /opt/ros/noetic/setup.bash
source /opt/ros/foxy/setup.bash
ros2 bag play -r 0.07 -s rosbag_v2 camera_2023-07-26-09-55-05_2.bag
```

Визуализировать результаты работы можно с помощью rviz
```
source /opt/ros/foxy/setup.bash
rviz2
```
