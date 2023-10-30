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
Для того, чтобы настроить CUDA Kernel для MSDeformAttn выполнить следующие команды:
```
cd oneformer/modeling/pixel_decoder/ops
sudo env "PATH=$PATH" sh make.sh
cd ../../../..
```
