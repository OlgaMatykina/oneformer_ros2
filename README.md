# oneformer_ros2

Создание образа из докер файла:
```
./build.sh
```

Запуск контейнера:
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
Затем запустить launch, который автоматически запустит необходимые компоненты:

```
ros2 launch semseg_ros2 semseg_launch.py
```

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
