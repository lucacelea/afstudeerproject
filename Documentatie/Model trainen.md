# Model trainen

Om ons eigen model te trainen zullen we gebruik maken van Tensorflow 2 en hun bijhorende TensorFlow 2 Object Detection API. Omdat trainen veel sneller gaat met het gebruik van een GPU zullen we ook Nvidia Cuda installeren zodat TensorFlow ten volle gebruik kan maken van de beschikbare GPU. Wij hebben deze install gedaan op Ubuntu 18.04.

[TOC]



### Cuda & CUDnn installeren

<u>Bij het installeren van Cuda is het zeer belangrijk om na te dat de versies van TensorFlow, Cuda en CUDnn compatibel zijn. Moest u zelf een andere versie willen installeren controleer dan zeker [*de lijst van TensorFlow zelf.*](https://www.tensorflow.org/install/source#gpu "Tensorflow Compatibility List")</u>



#### Cuda installeren

[Nvidia Cuda install Documentatie](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.htm l "Nvidia Documentatie")

Om Cuda te installeren kan je heel wat verschillende methodes gebruiken. Volgende methode gaf voor ons de meest succesvolle resultaten.

1. Download Cuda Toolkit 11.0  van de Nvidia website

   [Cuda Toolkit 11.0 Download](https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&amp;target_arch=x86_64&amp;target_distro=Ubuntu&amp;target_version=1804 "Cuda Toolkit 11.0") -->  Linux > x86_64 < Ubuntu < 18.04 < runfile (local) 

2. Als deze download gedaan is open je in Terminal window in de directory waar de runfile staat en voer je deze uit.

   <sub>Het opstarten van deze runfile kan soms wat lang duren </sub>

   ```bash
   sudo sh cuda_11.0.2_450.51.05_linux.run
   ```

   Volg deze install, wij installeren telkens alles behalve dat we de optie om een nieuwe Nvidia driver te installeren deselecteren. Dit zorgde vaak voor ons dat het fout loopt. Moest u toch een nieuwe Nvidia driver willen installeren doe je dit naar onze mening best afzonderlijk.

   Ook hier kan het soms wat langer duren maar als alles goed verloopt zou je als de installatie klaar is output moeten zien dat bevestigd dat de install correct is verlopen.

#### Cudnn installeren

[Nvidia cuDNN install Documentatie](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html "Nvidia cuDNN install Documentatie]")

1.  Download de cuDNN library uit de Archive

   [cuDNN Download](https://developer.nvidia.com/rdp/cudnn-download "cuDNN Download"), we downloaden de "[cuDNN Library for Linux (x86_64)](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.0_20201106/cudnn-11.0-linux-x64-v8.0.5.39.tgz)"

   <sub>Let goed op dat u de juiste cuDNN versie installeert, de versie dat wij gebruiken dat compatibel is met TensorFlow 2.4.0 en Cuda 11.0 is cuDNN 8.0.5. Deze is te vinden in de Archive.</sub>

![image-20210226130742952](/home/rafael/.config/Typora/typora-user-images/image-20210226130742952.png)

2.  Als de download gedaan is open je een Terminal in de directory waar de tgz file staat

   1. Unzip de cuDNN package

      ```bash
      tar -xzvf cudnn-11.0-linux-x64-v8.0.5.39.tgz
      ```

   2. Kopieer de volgende file naar de Cuda Toolkit directory

      ```bash
      sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
      sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
      sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
      ```



Als alle goed is verlopen zou u nu Cuda en cuDNN correct geinstalleerd hebben. Om u installatie te controleren kunt u altijd de Cuda Toolkit samples gebruiken. 

### TensorFlow Installeren

Voer het volgende command uit in een Terminal venster:

```bash
pip install --ignore-installed --upgrade tensorflow==2.4.0
```



