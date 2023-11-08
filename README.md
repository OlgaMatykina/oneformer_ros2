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
