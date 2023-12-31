Репозиторий создан для решения задачи семантической сегментации видеоизображения при движении по пешеходным зонам с выделением на них участков с дефектным дорожным покрытием (выбоины, ямы), тротуаров, асфальтированных дорог. Семантическая сегментация производится моделью OneFormer. Есть возможность проводить сегментацию двумя вариантами модели:  
а) обученная на 10 категориях:  
&emsp; &emsp;0: 'unlabelled',  
&emsp; &emsp;1: 'firehose',  
&emsp; &emsp;2: 'hose',  
&emsp; &emsp;3: 'waste',  
&emsp; &emsp;4: 'puddle',  
&emsp; &emsp;5: 'breakroad',  
&emsp; &emsp;6: 'sidewalk',  
&emsp; &emsp;7: 'terrain',  
&emsp; &emsp;8: 'vegetation',  
&emsp; &emsp;9: 'road'  
б) обученная на 6 категориях:  
&emsp; &emsp;0: 'unlabelled',  
&emsp; &emsp;1: 'breakroad',  
&emsp; &emsp;2: 'sidewalk',  
&emsp; &emsp;3: 'terrain',  
&emsp; &emsp;4: 'vegetation',  
&emsp; &emsp;5: 'road'  

Для визуализации результатов сегментации были выбраны следующие цвета:

0. 'unlabelled' : (255,255,255), # white
1. 'firehose' : (255,0,0), <font color=#FF0000># red </font>
2. 'hose' :(255,165,0), <font color=#FFA500># orange </font>
3. 'waste' :(0,0,255), <font color=#0000FF># blue </font>
4. 'puddle' :(255,255,0), <font color=#FFFF00># yellow </font>
5. 'breakroad' :(0,255,255), <font color=#00FFFF># aqua </font>
6. 'sidewalk' :(255,0,255), <font color=#FF00FF># magenta </font>
7. 'terrain' :(0,128,0), <font color=#008000># green </font>
8. 'vegetation' :(127,72,41), <font color=#7F4829># brown </font>
9. 'road' :(250,128,114) <font color=#FA8072># salmon </font>

# oneformer_ros2

Репозиторий содержит ROS2 (Foxy) интерфейс для работы с OneFormer.

Представленные инструкции позволяют собрать 2 узла:

- __oneformer_node__, который слушает топик с изображениями и отправляет результаты сегментации в топик segmentation_topic;
- __visualizer_node__, который слушает топики image и segmentation и визуализирует результаты сегментации, отправляя изображения в segmentation_color_topic.

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
После этого необходимо скачать веса по [ссылке](https://disk.yandex.ru/d/D2SmHXbhlHP5Uw) и поместить файл в папку 
```
~/oneformer_ros2/colcon_ws/src/semseg/weights
```
<!-- Затем нужно открыть конфигурационный файл, который расположен
```
~/oneformer_ros2/colcon_ws/src/semseg/weights/valid/swin/oneformer_swin_large_sem_seg_bs4_640k.yaml
```
и изменить название файла с весами (раскомментировать одну из строк):
```
  # WEIGHTS: /home/docker_oneformer_ros2/colcon_ws/src/semseg/weights/train1723_steps260k.pth
  # WEIGHTS: /home/docker_oneformer_ros2/colcon_ws/src/semseg/weights/train1723_steps210k.pth
```
-->
Затем запустить launch, который автоматически запустит необходимые компоненты, передав значения аргументов:
1. camera_ns
2. image_topic
3. cfg - принимает следующие значения:
    - home/docker_oneformer_ros2/colcon_ws/src/semseg/configs/config_6cats.yaml
    - home/docker_oneformer_ros2/colcon_ws/src/semseg/configs/config_10cats.yaml
4. cat_num - принимает значения 6 и 10

```
ros2 launch semseg_ros2 oneformer_launch.py camera_ns:="/sensum/left/" image_topic:=image_rect_color cfg:=/home/docker_oneformer_ros2/colcon_ws/src/semseg/configs/config_6cats.yaml cat_num:=6
```
Для тестирования работы узла нужно поместить ROS-bag в папку ~/oneformer_ros2/colcon_ws.
Для запуска проигрывания нужно сначала активировать окружение ROS1, затем ROS2:
```
cd ~/colcon_ws
source /opt/ros/noetic/setup.bash
source /opt/ros/foxy/setup.bash
ros2 bag play -r 0.07 -s rosbag_v2 camera_2023-07-26-09-55-05_2.bag
```
<!-- ros2 bag play -r 0.07 -s rosbag_v2 camera_2023-06-30-08-58-37_2.bag -->

Визуализировать результаты работы можно с помощью rviz
```
source /opt/ros/foxy/setup.bash
rviz2
```
