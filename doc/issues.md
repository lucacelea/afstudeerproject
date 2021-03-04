# Issues

Gedurende dit project hebben we veel technologieën uitgeprobeerd en getest. Om een ideale oplossing te vinden hebben we veel trail-and-error toegepast. In dit document gaan we de problemen aankaarten die we zijn tegengekomen tijdens het project.

# Inhoudstafel

- [Detecteren van dispenser](#detecteren-van-de-dispenser)
- [Detecteren van mensen](#detecteren-van-mensen)
- [Software stack](#software-stack)
    - [Cuda](#cuda)
    - [TensorRT](#tensorrt)
    - [Tensorflow](#tensorflow)
    - [ARM processor](#arm-processor)
    - [Jetpack](#jetpack)


## Detecteren van de dispenser
Voor het detecteren van de dispenser hebben we even moeten nadenken. Je kan dit op verschillende manieren proberen. Je kan een systeem training om een specifieke dispenser te herkennen. Alleen is dit niet zo efficiënt voor onze opdrachtgever. Telkens als de UZ Leuven een nieuwe dispenser zou gebruiken moet het systeem opnieuw getraind worden. Van dit idee stapte we dus al snel af. Ons volgende idee was om met een QR-code de dispenser te detecteren.

 In het begin leek dit een goede oplossing. We hadden een python script geschreven voor de Google Coral om, met behulp van Pyzbar (een extensie van python), QR-codes te kunnen detecteren. Vanaf hier begon het testen. We zorgden er eerst voor dat het systeem alleen de QR-code van de dispenser detecteerde en dit verliep ook vlot om te maken. Nu zaten we alleen nog met het probleem van afstand. We hadden dit op voorhand getest met een onze gsm’s. Hiermee konden we toch van vrij ver een kleine QR-code scannen. Alleen had de Google Coral het hier toch iets moeilijker mee. Aangezien de camera in de UZ Leuven op toch een vrij hoge en verre afstand van de dispenser zou hangen, besloten we deze na te bootsen. Bij het testen op deze situatie, konden we al snel concluderen dat we een zeer grote QR-code nodig hadden. Ook zou de detectie van de QR-code zeer afhankelijk zijn van in welke hoek deze naar de camera staat. We hebben ook de beelden van de camera omgezet met grayscalen. Dit zou het detecteren voor de camera makkelijker moeten maken. Te vergeefs heeft dit ook niet geholpen. Het idee van de QR-code detectie is dus moeilijker dan gedacht.
 
Het derde idee was om met een uniek logo te werken. Iets wat opvalt in het beeld van de camera. Opencv bied methodes tot het detecteren van specifieke images in een template, dus ons logo in het camera beeld. Wij hebben volgende methodes getest: TM_CCOEFF, TM_CCOEFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_SQDIFF, TM_SQDIFF_NORMED. Deze methodes bleken uiteindelijk niet performant genoeg te zijn. TM_CCOEFF en TM_CCOEFF_NORMED vonden het logo bijna altijd. Alleen trokken deze de detectie kader rond het logo veel te groot en waren dus niet nauwkeurig genoeg. De andere methodes vonden het logo bijna nooit. Dit idee had ook weer problemen met in welke hoek het logo naar de camera staat.

Het vierde idee was om met behulp van tekstrecognition de dispenser te detecteren. Aangezien de vorige zo moeilijk verliepen en na een korte meeting met de klant hadden we toch besloten om met een statische detectie zone te werken. De bedoeling is dat bij de installatie de gebruiker één keer een cirkel trekt op het camerabeeld. Als een gedetecteerd persoon in deze cirkel staat voor een voldoende aantal tijd, dan zal deze geteld worden als iemand die zijn handen heeft ontsmet. Deze laatste aanpak werkt zeer goed op ons systeem en we hebben vandaaruit het systeem verder opgebouwd.

## Detecteren van mensen
Voor de Jetson Xavier hebben we volgende projecten opgezet:

- https://github.com/dusty-nv/jetson-inference -> Deze werkte. Niet accuraat of performant genoeg. Weinig uitbreidingsmogelijkheden.

- https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3 -> Deze hebben we eerst getest op een desktop. Performantie en nauwkeurigheid was veelbelovend. Toen we het eindelijk draaiende kregen op de Jetson bleek er te weinig geheugen om het out of the box te draaien. De modellen moesten omgezet worden naar TensorRT modellen waardoor er zeer veel nauwkeurigheid verloren ging en de performantie was ook niet naar behoren. (1-3 frames per seconde)

 - https://github.com/ahmetozlu/tensorflow_object_counting_api -> Werkte niet op de Jetsons.

Voor de rest nog veel projecten uitgeprobeerd maar de meesten zijn niet accuraat of performant genoeg of er zijn problemen met compatibiliteit.
Dit waren allemaal projecten met TensorFlow. Aangezien deze niet werkten, zijn we in de richting van OpenCV beginnen zoeken.

Na wat zoeken hebben we uiteindelijk [een opensource broncode](https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/) gevonden voor het detecteren en tracken van mensen. Deze broncode gaf ons een goede start om er onze eigen versie van te maken, zo kon het al mensen detecteren, een id geven om te tracken doorheen de opeenvolgende frames en de gedetecteerde mensen tellen. In het begin telde deze code mensen die van boven naar onder en van onder naar boven wandelden in het beeld. Aangezien wij alleen de inkom moeten filmen hebben wij de code aangepast, zodat het alleen mensen die van onder naar boven wandelen telt. Ook hebben we een zone gemaakt waarin de gedetecteerde mensen worden geteld. Zo worden mensen die al geteld zijn geweest en van id zijn verandert niet opnieuw geteld, zolang ze uit deze zone staan.


## Software stack

### Cuda && cuDNN

Vooraleer we begonnen zijn met developpen op de Nvidia Jetson, moesten we (voor degelijke resultaten) Cuda installeren. De juiste methode hiervoor vonden we niet onmiddelijk, er kan veel fout lopen tijdens de installatie. Er bestaan vele methodes om de software te installeren en niet elke versie is compatibel met elke grafische kaart. Sommigen zijn volledig niet compatibel.

Na uitgebreid zoeken, hebben we de (voor ons) beste methode gevonden. Deze hebben we uitgeschreven in [onze guide voor het trainen van neurale netwerken](https://github.com/lucacelea/afstudeerproject/blob/main/doc/model-trainen.md#cuda--cudnn).

### Tensorflow

In het begin van het project wist elk teamlid nog niets over Tensorflow of toegepaste AI. Om Tensorflow werkende te krijgen op de Jetson hebben we verschillende bronnen gebruikt met gemengde resultaten. In 2019 is er een nieuwe (Tensorflow 2) versie uitgekomen die niet volledig backwards compatible is met Tensorflow 1. Dit zorgde voor veel problemen want de meeste projecten waren uitgewerkt in Tensorflow 1.

Er bestaat een compatibility mode voor Tensorflow 2 maar deze zorgde weer voor gemengde resultaten. De meeste (public) projecten gebruiken nog steeds Tensorflow 1 en dit zorgt dan weer voor andere versies van Cuda en cuDNN, etc. 

Uiteindelijk hebben we [een guide gevonden op de forums van Nvidia](https://forums.developer.nvidia.com/t/official-tensorflow-for-jetson-agx-xaviernx/141306) waar ze zelf Tensorflow hadden gebuild voor bepaalde versies van Jetpack.

Bovendien maken versie 1 en 2 van Tensorflow gebruik van andere soorten graphs namelijk Frozen Inference Graph en SavedModel Graph. Een Frozen Inference Graph kan niet meer verder worden getrained, terwijl een SavedModel dat wel kan (met de checkpoints).

### TensorRT 

Voor bepaalde Tensorflow modellen was er niet genoeg geheugen beschikbaar en moesten ze omgezet worden naar modellen die compatibel zijn met TensorRT. Hier zijn we uiteindelijk vanaf gestapt omdat ze dan veel minder accuraat werden.

Met het afstappen van het Tensorflow framework voor ons project op de Jetson hebben we dit niet tot in detail uitgepluisd. Mogelijks kan TensorRT dus wel een geschikte oplossing zijn voor dit project.

### Jetpack

Jetpack is de SDK die Nvidia aanbiedt waarop Ubuntu staat met hun specifiek gekozen software stack. (Een beetje zoals elke telefoonfabriekant hun eigen specifieke versie/skin van Android gebruikt) Voor de Jetson Xavier is hier weinig keuze omdat deze vrij nieuw is. Oudere versies van software zijn dus moeilijk te installeren. 

Voor de Xavier is er alleen de mogelijkheid voor Cuda 10.2. Deze downgraden is (zeer) moeilijk of zelfs onmogelijk, [aldus de Nvidia medewerkers op het forum](https://forums.developer.nvidia.com/t/i-want-to-downgrade-cuda-10-2-which-is-included-in-jetson-nano-by-default-to-10-0/140224). 

### OpenCV and numpy

Voor image processing wordt (vaak) OpenCV gebruikt dus besloten wij dit ook te doen. Bij het installeren moet er rekening mee gehouden worden dat OpenCV de dependency Numpy installeert. Numpy wordt ook gebruikt als dependency bij Tensorflow dus moet je zorgen dat je een OpenCV versie installeert dat gebruik maakt van een Numpy versie dat werkt met de geïnstalleerde Tensorflow versie. Voor ons was dat OpenCV versie 3.4.10.37, Numpy versie 1.16 en Tensorflow 1.15.
