

# Model trainen

Om ons eigen model te trainen zullen we gebruik maken van Tensorflow 2 en hun bijhorende TensorFlow 2 Object Detection API. Omdat trainen veel sneller gaat met het gebruik van een GPU zullen we voor ons project ook Nvidia Cuda gebruiken zodat TensorFlow ten volle gebruik kan maken van de beschikbare GPU. Wij hebben deze install gedaan op Ubuntu 18.04.

## Inhoudstafel

- [Model trainen](#model-trainen)
  * [Cuda & cuDNN](#cuda---cudnn)
    + [Cuda installeren](#cuda-installeren)
    + [Cudnn installeren](#cuDNN-installeren)
  * [TensorFlow](#tensorflow)
    + [Installeren](#installeren)
  * [TensorFlow Object Detection API](#tensorflow-object-detection-api)
    + [TensorFlow Model Garden downloaden](#tensorflow-model-garden-downloaden)
    + [Protobuf Installatie/ Compilatie](#protobuf-installatie--compilatie)
    + [COCO API installatie](#coco-api-installatie)
  * [Training Custom Object Detector](#training-custom-object-detector)
    + [De <em>Workspace</em> voorbereiden](#de--em-workspace--em--voorbereiden)
    + [De Dataset voorbereiden](#de-dataset-voorbereiden)
      - [De Dataset annoteren](#de-dataset-annoteren)
      - [De Dataset partitioneren](#de-dataset-partitioneren)
    + [De Label Map aanmaken](#de-label-map-aanmaken)
    + [TensorFlow records aanmaken](#tensorflow-records-aanmaken)
      - [*.xml omzetten naar *.record](#-code--xml--code--omzetten-naar--code--record--code-)
    + [Training taak aanmaken](#training-taak-aanmaken)
      - [Download Pre-Trained Model](#download-pre-trained-model)
      - [De Training Pipeline configureren](#de-training-pipeline-configureren)
    + [Het model trainen](#het-model-trainen)
    + [Getrained model extraheren](#getrained-model-extraheren)
- [Bronnen](#bronnen)

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

### cuDNN installeren

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

### De Label Map aanmaken

TensorFlow heeft een label map nodig, die namelijk elk van de gebruikte labels in een integer waarde omzet. Deze label map wordt zowel door het training- als het detectieproces gebruikt.

Volgende Label Map hebben wij gebruikt.

```
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


### TensorFlow records aanmaken

Nu onze annotaties zijn gegenereerd en onze dataset zijn opgesplitst in de gewenste training en testing subsets, is het tijd om onze annotaties om te zetten in het zogenaamde <code>TFRecord</code> formaat.

#### <code>*.xml</code> omzetten naar <code>*.record</code>

Om dit te doen kan een scriptje gebruikt wordt dat itereert over alle <code>*.xml</code> bestanden in de <code>training_demo/images/train</code> en <code>training_demo/images/test</code> mappen en bijhorende <code>*.record</code> bestanden aanmaakt. Gelukkig heeft iemand dit al voor ons gedaan. [Dit scriptje is hier te downloaden.](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_downloads/da4babe668a8afb093cc7776d7e630f3/generate_tfrecord.py "Scriptje om *.xml om te zetten naar de *.record bestanden")
Plaats dit scriptje in de <code>Tensorflow/scripts/preprocessing</code> map.

Dit scriptje heeft <code>pandas</code> als dependency dus die moet geinstalleerd worden.
```bash
pip3 install pandas
```

Vervolgens in de <code>TensorFlow/scripts/preprocessing</code> map:

```bash
# Training data aanmaken:
python3 generate_tfrecord.py -x [PAD_NAAR_IMAGES_MAP]/train -l [PAD_NAAR_ANNOTATIONS_MAP]/label_map.pbtxt -o [PAD_NAAR_ANNOTATIONS_MAP]/train.record

# Test data aanmaken:
python3 generate_tfrecord.py -x [PAD_NAAR_IMAGES_MAP]/test -l [PAD_NAAR_ANNOTATIONS_MAP]/label_map.pbtxt -o [PAD_NAAR_ANNOTATIONS_MAP]/test.record
```

Het uitvoeren van dit zou moeten resulteren in 2 new bestanden onder de <code>training_demo/annotations</code> map, genaamd <code>test.record</code> en <code>training.record</code>.


### Training taak aanmaken

Wij zullen geen taining taak aanmaken vanaf nul maar opteren eerder om een bestaand model te gebruiken. Moest u dit toch willen doen kunt u [de documentatie van Tensorflow lezen.](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md)

Het model dat we in onze voorbeelden zullen gebruiken is het [SSD ResNet50 V1 FPN 640x640](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) model, omdat het een relatief goede afweging biedt tussen prestaties en snelheid. Er bestaan echter een aantal andere modellen die je kunt gebruiken, die allemaal opgesomd staan in [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).

#### Download Pre-Trained Model

Om te beginnen download een model uit de lijst vermeld hierboven.

Eenmaal het <code>*.tar.gz</code> bestand is gedownload pak je de ze uit in de <code>training_demo/pre-trained-models</code> map.
Dit zou moeten resulteren in een gelijkaardige structuur als het volgende:

```
training_demo/
├─ ...
├─ pre-trained-models/
│  └─ ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
└─ ...
```

#### De Training Pipeline configureren

Nu het model dat gebruikt zal worden is gedownload maken we in de een nieuwe <code>my_ssd_resnet50_v1_fpn</code> map aan voor ons eigen getrained model. Dit doen we in de <code>training_demo/models</code> map. Hiernaar Kopiëren we de <code>pipeline.config</code> dat we terugvinden in ons gedownload model.

```
training_demo/
├─ ...
├─ models/
│  └─ my_ssd_resnet50_v1_fpn/
│     └─ pipeline.config
└─ ...
```

In deze <code>pipeline.config</code> zijn een aantal aanpassingen nodig. (Deze aanpassingen zijn analoog voor andere modellen).


<details>
  <summary>Uitvouwen</summary>


```
model {
  ssd {
    num_classes: 2 # Pas dit aan naargelang het aantal klassen dat nodig zijn (label_map.pbtxt)
    image_resizer {
      fixed_shape_resizer {
        height: 640
        width: 640
      }
    }
    feature_extractor {
      type: "ssd_resnet50_v1_fpn_keras"
      depth_multiplier: 1.0
      min_depth: 16
      conv_hyperparams {
        regularizer {
          l2_regularizer {
            weight: 0.00039999998989515007
          }
        }
        initializer {
          truncated_normal_initializer {
            mean: 0.0
            stddev: 0.029999999329447746
          }
        }
        activation: RELU_6
        batch_norm {
          decay: 0.996999979019165
          scale: true
          epsilon: 0.0010000000474974513
        }
      }
      override_base_feature_extractor_hyperparams: true
      fpn {
        min_level: 3
        max_level: 7
      }
    }
    box_coder {
      faster_rcnn_box_coder {
        y_scale: 10.0
        x_scale: 10.0
        height_scale: 5.0
        width_scale: 5.0
      }
    }
    matcher {
      argmax_matcher {
        matched_threshold: 0.5
        unmatched_threshold: 0.5
        ignore_thresholds: false
        negatives_lower_than_unmatched: true
        force_match_for_each_row: true
        use_matmul_gather: true
      }
    }
    similarity_calculator {
      iou_similarity {
      }
    }
    box_predictor {
      weight_shared_convolutional_box_predictor {
        conv_hyperparams {
          regularizer {
            l2_regularizer {
              weight: 0.00039999998989515007
            }
          }
          initializer {
            random_normal_initializer {
              mean: 0.0
              stddev: 0.009999999776482582
            }
          }
          activation: RELU_6
          batch_norm {
            decay: 0.996999979019165
            scale: true
            epsilon: 0.0010000000474974513
          }
        }
        depth: 256
        num_layers_before_predictor: 4
        kernel_size: 3
        class_prediction_bias_init: -4.599999904632568
      }
    }
    anchor_generator {
      multiscale_anchor_generator {
        min_level: 3
        max_level: 7
        anchor_scale: 4.0
        aspect_ratios: 1.0
        aspect_ratios: 2.0
        aspect_ratios: 0.5
        scales_per_octave: 2
      }
    }
    post_processing {
      batch_non_max_suppression {
        score_threshold: 9.99999993922529e-09
        iou_threshold: 0.6000000238418579
        max_detections_per_class: 100
        max_total_detections: 100
        use_static_shapes: false
      }
      score_converter: SIGMOID
    }
    normalize_loss_by_num_matches: true
    loss {
      localization_loss {
        weighted_smooth_l1 {
        }
      }
      classification_loss {
        weighted_sigmoid_focal {
          gamma: 2.0
          alpha: 0.25
        }
      }
      classification_weight: 1.0
      localization_weight: 1.0
    }
    encode_background_as_zeros: true
    normalize_loc_loss_by_codesize: true
    inplace_batchnorm_update: true
    freeze_batchnorm: false
  }
}
train_config {
  batch_size: 4 # Verhoog / Verlaag deze waarde afhankelijk van het beschikbare geheugen (Hogere waarden vereisen meer geheugen en vice-versa). Hiermee kan wat gespeeld worden, 4 was comfortabel voor een GTX 1080
  data_augmentation_options {
    random_horizontal_flip {
    }
  }
  data_augmentation_options {
    random_crop_image {
      min_object_covered: 0.0
      min_aspect_ratio: 0.75
      max_aspect_ratio: 3.0
      min_area: 0.75
      max_area: 1.0
      overlap_thresh: 0.0
    }
  }
  sync_replicas: true
  optimizer {
    momentum_optimizer {
      learning_rate {
        cosine_decay_learning_rate {
          learning_rate_base: 0.03999999910593033
          total_steps: 25000
          warmup_learning_rate: 0.013333000242710114
          warmup_steps: 2000
        }
      }
      momentum_optimizer_value: 0.8999999761581421
    }
    use_moving_average: false
  }
  fine_tune_checkpoint: "pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0" # Pad naar de checkpoint van het pre-trained model
  num_steps: 25000
  startup_delay_steps: 0.0
  replicas_to_aggregate: 8
  max_number_of_boxes: 100
  unpad_groundtruth_tensors: false
  fine_tune_checkpoint_type: "detection" # Zet dit naar "detection" omdat we het volledig detectie model willen trainen
  use_bfloat16: false # Set dit naar false als je niet traint op een TPU
  fine_tune_checkpoint_version: V2
}
train_input_reader {
  label_map_path: "annotations/label_map.pbtxt" # Pad naar label map bestand
  tf_record_input_reader {
    input_path: "annotations/train.record" # Pad naar training TFRecord bestand
  }
}
eval_config {
  metrics_set: "coco_detection_metrics"
  use_moving_averages: false
}
eval_input_reader {
  label_map_path: "annotations/label_map.pbtxt" # Pad naar label map bestand
  shuffle: false
  num_epochs: 1
  tf_record_input_reader {
    input_path: "annotations/test.record" # Pad naar testing TFRecord
  }
}
```

</details>

<sub>Controleer zeker al paden dat meegegeven moeten worden, soms is een absoluut pad veiliger en vermijdt dit fouten.</sub>

### Het model trainen

Voordat we beginnen met het trainen van ons model, kopiëren we het <code>TensorFlow/models/research/object_detection/model_main_tf2.py</code> script en plakken het rechtstreeks in onze <code>training_demo map</code>. We zullen dit script nodig hebben om ons model te trainen.


Om het trainen te starten voer je dit script uit:

```bash
python3 model_main_tf2.py --model_dir=models/my_ssd_resnet50_v1_fpn --pipeline_config_path=models/my_ssd_resnet50_v1_fpn/pipeline.config
```

Als alles goed verloopt zou gelijkaardig output moeten verschijnen eenmaal het trainen is begonnen.

```
...
WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.gamma
W0716 05:24:19.105542  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.gamma
WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.beta
W0716 05:24:19.106541  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.beta
WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_mean
W0716 05:24:19.107540  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_mean
WARNING:tensorflow:Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_variance
W0716 05:24:19.108539  1364 util.py:143] Unresolved object in checkpoint: (root).model._box_predictor._base_tower_layers_for_heads.class_predictions_with_background.4.10.moving_variance
WARNING:tensorflow:A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
W0716 05:24:19.108539  1364 util.py:151] A checkpoint was restored (e.g. tf.train.Checkpoint.restore or tf.keras.Model.load_weights) but not all checkpointed values were used. See above for specific issues. Use expect_partial() on the load status object, e.g. tf.train.Checkpoint.restore(...).expect_partial(), to silence these warnings, or use assert_consumed() to make the check explicit. See https://www.tensorflow.org/guide/checkpoint#loading_mechanics for details.
WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
INFO:tensorflow:Step 100 per-step time 1.153s loss=0.761
I0716 05:26:55.879558  1364 model_lib_v2.py:632] Step 100 per-step time 1.153s loss=0.761
...
```

Ter controle dat de GPU wel degelijk wordt gebruikt om te trainen kan in een andere terminal volgend commando uitgevoerd worden. Met dit is het mogelijk te controleren hoeveel resources worden gebruikt. Dit kan ook handig zijn als u de <code>batch_size</code> wilt aanpassen.

```bash
watch nvidia-smi
```
Nu zal het even wachten zijn voordat het model volledig getrained is.

Volgends van wat mensen online hebben gezegd, lijkt het erop dat het raadzaam is uw model een TotalLoss van minstens 2 (idealiter 1 en lager) te laten bereiken als u "eerlijke" detectieresultaten wilt bereiken. Uiteraard is een lager TotalLoss beter, maar een zeer laag TotalLoss moet worden vermeden, omdat het model uiteindelijk de dataset kan overfitten, wat betekent dat het slecht zal presteren wanneer het wordt toegepast op afbeeldingen buiten de dataset.

Om het trainen overzichterlijk te monitoren kunt gebruik gemaakt worden van [TensorBoard](https://www.tensorflow.org/tensorboard "TensorBoard")

Open een terminal in <code>training_demo</code> en voer volgend commando uit:

```bash
tensorboard --logdir=models/my_ssd_resnet50_v1_fpn
```

Het bovenstaande commando zal een nieuwe TensorBoard server starten, die (standaard) luistert naar poort 6006 van uw machine. Ervan uitgaande dat alles goed is gegaan, zou u een output moeten zien zoals hieronder (plus/minus enkele waarschuwingen):
```
...
TensorBoard 2.2.2 at http://localhost:6006/ (Press CTRL+C to quit)
```

Zodra dit is gebeurd, gaat u naar uw browser en typt u http://localhost:6006/ in de adresbalk, waarna u een dashboard te zien zou moeten krijgen dat lijkt op het hieronder getoonde (misschien minder bevolkt als uw model nog maar net met trainen is begonnen):

![TensorBoard Dashboard](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/_images/TensorBoard.JPG)


### Getrained model extraheren

Eenmaal u tevreden bent met hoe uw model getrained is kunt u deze exporteren. Nu zullen we de nieuw getrainde <em>inference graph</em> extraheren. Dit doet men als volgt.

Kopieer het <code>TensorFlow/models/research/object_detection/exporter_main_v2.py</code> script en plak het rechtstreeks in je <code>training_demo</code> map.
Open nu een terminal in deze map en voer het volgende commando uit.

```bash
python3 .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\my_efficientdet_d1\pipeline.config --trained_checkpoint_dir .\models\my_efficientdet_d1\ --output_directory .\exported-models\my_model
```

Na dit process voltooid is zou uw model klaar moeten zijn voor gebruik, deze zou u moeten terugvinden in de map <code>training_demo/exported-models</code>.

```
training_demo/
├─ ...
├─ exported-models/
│  └─ my_model/
│     ├─ checkpoint/
│     ├─ saved_model/
│     └─ pipeline.config
└─ ...
```

Met dit model kunt u nu inferentie doen.


___

# Bronnen
* [Nvidia cuDNN Documentation](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html "Nvidia cuDNN Documentation")
* [Cuda Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html "Cuda Toolkit Documentation")
* [TensorFlow 2 Object Detection API tutorial Installation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html "TensorFlow 2 Object Detection API tutorial Installation")
* [TensorFlow 2 Object Detection API tutorial Training Custom Object Detector](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html "TensorFlow 2 Object Detection API tutorial Training Custom Object Detector")



