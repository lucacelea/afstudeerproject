

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

![image-20210226130742952](https://github.com/lucacelea/afstudeerproject/blob/main/doc/docs_images/model%20trainen/image-20210226130742952.png)

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
pip3 install --ignore-installed --upgrade tensorflow==2.4.0
```

Volgend commando kan gebruikt worden om te testen of de installatie succesvol was

```bash
python3 -c 'import tensorflow as tf; print(tf.__version__)'
```

Als dat commando succesvol was, kan nagegaan worden of de installatie samen werkt met Tensorflow. Dit door volgend commando te gebruiken.

```bash
python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

Als dit succesvol was kunt u ook nagaan of Tensorflow werkt met de Cuda installatie dat we daarnet hebben gedaan

```bash
python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

Dit commando zou ouptut moeten genereren gelijkaardig aan het volgende


<details>
  <summary>Uitvouwen!</summary>
  
```
2020-06-22 20:24:31.355541: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
2020-06-22 20:24:33.650692: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library nvcuda.dll
2020-06-22 20:24:33.686846: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties:
pciBusID: 0000:02:00.0 name: GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 8.00GiB deviceMemoryBandwidth: 238.66GiB/s
2020-06-22 20:24:33.697234: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
2020-06-22 20:24:33.747540: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_10.dll
2020-06-22 20:24:33.787573: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cufft64_10.dll
2020-06-22 20:24:33.810063: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library curand64_10.dll
2020-06-22 20:24:33.841474: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusolver64_10.dll
2020-06-22 20:24:33.862787: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusparse64_10.dll
2020-06-22 20:24:33.907318: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-06-22 20:24:33.913612: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
2020-06-22 20:24:33.918093: I tensorflow/core/platform/cpu_feature_guard.cc:143] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
2020-06-22 20:24:33.932784: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2382acc1c40 initialized for platform Host (this does not guarantee that XLA will be used). Devices:
2020-06-22 20:24:33.939473: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): Host, Default Version
2020-06-22 20:24:33.944570: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1561] Found device 0 with properties:
pciBusID: 0000:02:00.0 name: GeForce GTX 1070 Ti computeCapability: 6.1
coreClock: 1.683GHz coreCount: 19 deviceMemorySize: 8.00GiB deviceMemoryBandwidth: 238.66GiB/s
2020-06-22 20:24:33.953910: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudart64_101.dll
2020-06-22 20:24:33.958772: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cublas64_10.dll
2020-06-22 20:24:33.963656: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cufft64_10.dll
2020-06-22 20:24:33.968210: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library curand64_10.dll
2020-06-22 20:24:33.973389: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusolver64_10.dll
2020-06-22 20:24:33.978058: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cusparse64_10.dll
2020-06-22 20:24:33.983547: I tensorflow/stream_executor/platform/default/dso_loader.cc:44] Successfully opened dynamic library cudnn64_7.dll
2020-06-22 20:24:33.990380: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1703] Adding visible gpu devices: 0
2020-06-22 20:24:35.338596: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1102] Device interconnect StreamExecutor with strength 1 edge matrix:
2020-06-22 20:24:35.344643: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1108]      0
2020-06-22 20:24:35.348795: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1121] 0:   N
2020-06-22 20:24:35.353853: I tensorflow/core/common_runtime/gpu/gpu_device.cc:1247] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 6284 MB memory) -> physical GPU (device: 0, name: GeForce GTX 1070 Ti, pci bus id: 0000:02:00.0, compute capability: 6.1)
2020-06-22 20:24:35.369758: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0x2384aa9f820 initialized for platform CUDA (this does not guarantee that XLA will be used). Devices:
2020-06-22 20:24:35.376320: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecutor device (0): GeForce GTX 1070 Ti, Compute Capability 6.1
tf.Tensor(122.478485, shape=(), dtype=float32)
```

</details>



