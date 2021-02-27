

# Model trainen

Om ons eigen model te trainen zullen we gebruik maken van Tensorflow 2 en hun bijhorende TensorFlow 2 Object Detection API. Omdat trainen veel sneller gaat met het gebruik van een GPU zullen we voor ons project ook Nvidia Cuda gebruiken zodat TensorFlow ten volle gebruik kan maken van de beschikbare GPU. Wij hebben deze install gedaan op Ubuntu 18.04.

[TOC]

## Cuda & cuDNN 

Bij het installeren van Cuda is het zeer belangrijk om na te dat de versies van TensorFlow, Cuda en CUDnn compatibel zijn. Moest u zelf een andere versie willen installeren controleer dan zeker [*de lijst van TensorFlow zelf.*](https://www.tensorflow.org/install/source#gpu "Tensorflow Compatibility List")

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

Als dat commando succesvol was, kan nagegaan worden of de Cuda installatie van daarnet samenwerkt met TensorFlow. Dit door volgend commando te gebruiken.

```bash
python3 -c "import tensorflow as tf;print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```

Dit commando zou ouptut moeten genereren gelijkaardig aan het volgende

<details>
  <summary>Uitvouwen</summary>


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

Controleer zeker dat er nergens in de output meldingen zijn van errors en dat de beschikbare GPU wel degelijk getoond wordt.

## TensorFlow Object Detection API


### TensorFlow Model Garden downloaden

Maak ergens een nieuwe map genaamd TensorFlow aan. (e.g. /home/uw_gebruiker/TensorFlow).

Open vervolgens in deze nieuwe map een terminal en clone de [TensorFlow Models repository](https://github.com/tensorflow/models "Tensorflow Models repository")

Dit zou moeten resulteren in volgende folder structuur.

```
TensorFlow/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   ├─ research/
   └── ...
```

### Protobuf Installatie/ Compilatie

De Tensorflow Object Detection API gebruikt Protobuf om model- en trainingsparameters te configureren. Voordat het framework kan worden gebruikt, moeten de Protobuf libraries worden gedownload en gecompileerd.

To-Do weet dit niet meer vanbuiten en was dacht ik anders dan wat er in de documentatie stond

### COCO API installatie

Vanaf TensorFlow 2.x, wordt de <em>pycocotools</em> package opgelijst als een dependency van de Object Detectie API. Idealiter zou dit pakket geïnstalleerd moeten worden wanneer de Object Detection API geïnstalleerd wordt, zoals gedocumenteerd in de installatie van de Object Detection API in de sectie hieronder, maar de installatie kan om verschillende redenen mislukken en daarom is het eenvoudiger om het pakket gewoon vooraf te installeren, in welk geval de latere installatie overgeslagen zal worden.


Clone de [cocoapi repository](https://github.com/cocodataset/cocoapi "cocoapi repository") en dan <code>make</code> en kopieer de pycocotools subfolder naar de <code>Tensorflow/models/research</code> map als volgt.


```bash
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
make
cp -r pycocotools <PATH_TO_TF>/TensorFlow/models/research/
```

###Installeer de Object Detection API

De installatie van de Object Detection API wordt behaald door het installeren van de <code>object_detection</code> package. Dit wordt gedaan door volgende commando's uit te voeren vanuit de <code>Tensorflow/models/research</code> folder:

```bash
cp object_detection/packages/tf2/setup.py .
python3 -m pip install .
```

Als alles goed is verlopen en je krijgt geen errors kan nu de installatie worden getest door het volgende commando uit te voeren vanuit de <code>Tensorflow/models/research</code> map:

```bash
python3 object_detection/builders/model_builder_tf2_test.py
```

Als dit resulteert in volgende output dan is alles goed verlopen.

```
...
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 20 tests in 68.510s

OK (skipped=1)
```



## Training Custom Object Detector


### De <em>Workspace</em> voorbereiden

Als alles goed verlopen is zou nu onder de <code>Tensorflow</code> map volgende structuur terug te vinden zijn.

```
TensorFlow/
├─ addons/ (Optional)
│  └─ labelImg/
└─ models/
   ├─ community/
   ├─ official/
   ├─ orbit/
   ├─ research/
   └─ ...
```

Om alles gestructureerd op te zetten zullen we in deze map nog wat extra mappen aanmaken.
Maak onder de <code>Tensorflow</code> een map genaamd <code>workspace</code>.
Maak in deze nieuwe map vervolgens ook een folder genaamd <code>training_demo</code>.
Dit zou moeten resulteren in volgende structuur.

```
TensorFlow/
├─ addons/ (Optional)
├─ models/
│  ├─ community/
│  ├─ official/
│  ├─ orbit/
│  ├─ research/
│  └─ ...
└─ workspace/
   └─ training_demo/
```

De training_demo map is onze trainingsmap, die alle bestanden zal bevatten die te maken hebben met de training van ons model. Het is aangeraden om een aparte trainingsmap aan te maken telkens als we op een andere dataset willen trainen. De typische structuur voor trainingsmappen is hieronder weergegeven.

```
training_demo/
├─ annotations/
├─ exported-models/
├─ images/
│  ├─ test/
│  └─ train/
├─ models/
├─ pre-trained-models/
```
Deze mappen kunnen nu al aangemaakt worden. Het doel van elke map zal wel snel duidelijk worden

### De Dataset voorbereiden

#### De Dataset annoteren

Om de dataset te anntoren zullen we gebruiken maken van de <code>labelImg</code>.

Installeer simpelweg met <code>PIP</code> de package.

```bash
pip3 install labelImg
```

Hierna kan <code>labelImg</code> opgestart worden door het volgende in de terminal uit te voeren:

```bash
labelImg
```

Verzamel foto's waarmee het model getrained zal worden, 100 of meer wordt aangeraden. Plaats deze in de <code>training_demo/images</code>. Deze foto's zullen we annoteren zodat het model hierop kan trainen.


In <code>labelImg</code> kan u nu deze folder openen en zou u rechtsonder alle foto's moeten opgelijst worden.

Nu kan er begonnen worden met annoteren. Probeer steeds zo dicht mogelijk rond het object te selecteren. Bij elke foto moet u deze opslaan als alles geannoteerd is dat u nodig hebt. Dit zal een bijhorende <code>*.xml</code> bestand aanmaken.

![labelImg](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_images/labelImg.JPG)

#### De Dataset partitioneren

Vaak wordt na het annoteren van de foto's de data gepartitioneerd. De foto's en bijhorende <code>*.xml</code> bestanden worden opgedeeld in een training-set en test-set. Met de training-set wordt het model getrained, de test-set wordt dan weer gebruikt om te evalueren hoe goed het model.

Een aangeraden verdeling is 90% training en 10% test.

Plaats na te partitioneren de foto's (en hun bijhorende <code>*.xml</code> betanden) in bijhorende subfolder:
<code>training_demo/images/train</code>
<code>training_demo/images/test</code>

#### De Label Map aanmaken

TensorFlow heeft een label map nodig, die namelijk elk van de gebruikte labels in een integer waarde omzet. Deze label map wordt zowel door het training- als het detectieproces gebruikt.

Volgende Label Map hebben wij gebruikt.

```json
item {
    id: 1
    name: 'person'
}

item {
    id: 2
    name: 'dispenser'
}
```

Label map bestanden hebben gewoonlijk de extensie <code>.pbtxt</code> en horen in de <code>training_demo/annotations</code> map.


