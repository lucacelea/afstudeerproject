# Model trainen

Om ons eigen model te trainen zullen we gebruik maken van Tensorflow 2 en hun bijhorende TensorFlow 2 Object Detection API. Omdat trainen veel sneller gaat met het gebruik van een GPU zullen we voor ons project ook Nvidia Cuda gebruiken zodat TensorFlow ten volle gebruik kan maken van de beschikbare GPU. Wij hebben deze install gedaan op Ubuntu 18.04.

[TOC]

## Cuda & CUDnn installeren

<u>Bij het installeren van Cuda is het zeer belangrijk om na te dat de versies van TensorFlow, Cuda en CUDnn compatibel zijn. Moest u zelf een andere versie willen installeren controleer dan zeker [*de lijst van TensorFlow zelf.*](https://www.tensorflow.org/install/source#gpu "Tensorflow Compatibility List")</u>

### Cuda installeren

Om Cuda werkende te krijgen, kan er gebruik gemaakt worden van verschillende guides. [Nvidia heeft documentatie voor de installatie van Cuda](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html "Nvidia Documentatie") die hierbij kan helpen. Volgende methode gaf voor ons project de meest succesvolle resultaten.

Download de [Cuda Toolkit 11.0 Download](https://developer.nvidia.com/cuda-11.0-download-archive?target_os=Linux&amp;target_arch=x86_64&amp;target_distro=Ubuntu&amp;target_version=1804 "Cuda Toolkit 11.0") van de Nvidia website. Kies voor een geschikte architectuur.

      Voor ons project kozen we:  Linux > x86_64 < Ubuntu < 18.04 < runfile (local) 

Voer nadien de runfile uit.

```bash
sudo sh cuda_11.0.2_450.51.05_linux.run
```
<sub>Het opstarten van deze runfile kan lang duren </sub>

Volg deze install. Installeer alles buiten de optie om een nieuwe Nvidia driver te installeren. Deze optie kan voor problemen zorgen. Na de installatie is er een bevestiging dat alles vlot verlopen is.

### Cudnn installeren

Voor de installatie van cuDNN [is er ook documentatie van Nvidia](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html "Nvidia cuDNN install Documentatie]"). Download de [cuDNN Library](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.0.5/11.0_20201106/cudnn-11.0-linux-x64-v8.0.5.39.tgz) uit de archive.

![image-20210226130742952](https://github.com/lucacelea/Afstudeerproject/blob/main/doc/docs_images/model%20trainen/image-20210226130742952.png)

<sub>Let op de versie van cuDNN is belangrijk. De versie dat wij gebruiken is compatibel is met TensorFlow 2.4.0 en Cuda 11.0. (cuDNN 8.0.5)</sub>

Unzip de cuDNN package

```bash
tar -xzvf cudnn-11.0-linux-x64-v8.0.5.39.tgz
```

Kopieer de volgende file naar de Cuda Toolkit directory

```bash
sudo cp cuda/include/cudnn*.h /usr/local/cuda/include 
sudo cp -P cuda/lib64/libcudnn* /usr/local/cuda/lib64 
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

Cuda en cuDNN zouden nu correct geinstalleerd moeten zijn. Om de installatie te controleren kunnen de Cuda Toolkit samples gebruikt worden. 

## TensorFlow

### Installeren

Voer het volgende command uit in een Terminal venster:

```bash
pip install --ignore-installed --upgrade tensorflow==2.4.0
```

Volgend commando kan gebruikt worden om te testen of de installatie succesvol was

```
python3 -c 'import tensorflow as tf; print(tf.__version__)'  # for Python 3
```

Als dat commando succesvol was, kan nagegaan worden of de installatie samen werkt met Tensorflow. Dit door volgend commando te gebruiken.

```bash
python -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
