# Model trainen

Om ons eigen model te trainen zullen we gebruik maken van Tensorflow 2 en hun bijhorende TensorFlow 2 Object Detection API. Omdat trainen veel sneller gaat met het gebruik van een GPU zullen we ook Nvidia Cuda installeren zodat TensorFlow ten volle gebruik kan maken van de beschikbare GPU.

##### Cuda & CUDnn installeren

[https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html]: 

<u>Bij het installeren van Cuda is het zeer belangrijk om na te dat de versies van TensorFlow, Cuda en CUDnn compatibel zijn. Moest u zelf een andere versie willen installeren controleer dan zeker de lijst van TensorFlow zelf.</u>

https://www.tensorflow.org/install/source#gpu



##### TensorFlow Installeren

Voer het volgende command uit in een Terminal venster:

```bash
pip install --ignore-installed --upgrade tensorflow==2.4.1
```



