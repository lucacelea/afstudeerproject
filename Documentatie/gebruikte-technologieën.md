# OpenCV

OpenCV (Open Source Computer Vision Library) is een open source computer vision en machine learning library. Ze is gemaakt om programmeurs een gemeenschappelijke infrastructuur aan te bieden om ontwikkeling van (commerciële) computer vision en machine learning applicaties te vergemakkelijken. Doordat OpenCV gebruik maakt van de BSD licentie is het zeer toegankelijk voor bedrijven en programmeurs om de software te gebruiken en aan te passen. BSD licenties behoren namelijk tot de familie van de permissive free software licenses. Dit betekent dat er minimale restricties zijn voor het gebruiken en aanpassen van de software. 

Deze bibliotheek maakt gebruik van meer dan 2500 algoritmes, waaronder klassieke en state-of-the-art computer vision en machine learning algoritmes. Deze algoritmes kunnen gebruikt worden om onder meer gezichten te herkennen en detecteren, objecten te identificeren, menselijke acties te classificiëren, camera bewegingen volgen, 3D modellen extraheren uit objecten, afbeeldingen samenbrengen tot een groter geheel, soortgelijke afbeeldingen in een database zoeken, oogbeweging volgen, etc. OpenCV heeft een grote community en veel hoogstaande bedrijven die de library gebruiken. (Google, Yahoo, IBM, Sony, etc.)

# Computer vision

Computer vision is een type van artificiële intelligentie. Het is een technologie die er zich op focust om objecten te herkennen in een afbeelding. Hierbij leent het delen van de complexiteit van het menselijk zichtsysteem. Dit maakt het mogelijk voor computers om objecten in een afbeelding of video te herkennen net zoals mensen dat kunnen. 

Dankzij vooruitgang in de artificiële intelligentie en innovaties in deep learning en neurale netwerken zijn er computer vision systemen die performanter zijn dan menselijke systemen. Sommige systemen hebben een performantie van 99 procent. 

Computer visie baseert zich op pattern recognition. Een manier om een computer te trainen om afbeeldingen te kunnen herkennen is om deze zeer veel afbeeldingen te geven die gelabeld zijn. Daarna worden deze afbeeldingen geanalyseerd met bepaalde algoritmes zodat de computer een idee krijgt van wat een bepaald object echt is. Hierna is het in theorie mogelijk dat de computer ongelabelde afbeeldingen zou moeten kunnen herkennen.

# Imutils

Imutils is een library die werken met OpenCV vergemakkelijkt. De library voorziet vijf hoofdfuncties.

-	Rotatie
Maakt het mogelijk om afbeeldingen te roteren met een bepaalde hoek.

-	Resizing
Kan de grootte van de afbeelding aanpassen in pixels.

-	Skeletonization
Kan de structuur van de afbeelding afbeelden gegeven dat de afbeelding zwart-wit is.

-	Display met Matplotlib
Opent afbeeldingen in RBG formaat in plaats van de OpenCV standaard BGR.

-	Translatie
Kan afbeeldingen transleren via bepaalde assen over bepaalde lengtes.

# Neurale netwerken

Een neuraal netwerken kan een “echt” biologisch neuraal netwerk zijn, zoals in een menselijk brein, of het kan kunstmatig gesimuleerd worden in een computer. Om zulke gesimuleerde neurale netwerken te snappen moeten we eerst een aantal begrippen bespreken. 

Allereerst het begrip “Machine learning”. Machine learning is een vorm van artificiële intelligentie die gericht op het bouwen van systemen die van de verwerkte data kunnen leren of data gebruiken om beter te presteren. Bijvoorbeeld: men voedt een programma tientalle duizenden foto’s waarin aangeduidt wordt wat een fiets is. Met deze informatie kan het programma voorspellen of er in een gegeven andere foto een fiets staat of niet. 

Vervolgens het begrip “Deep learning”. Deep learning verwijst naar bepaalde soorten technieken voor machinaal leren waarin verschillende “lagen” eenvoudige verwerkingseenheden in een netwerk zijn verbonden, zodat de input in het systeem achtereenvolgens door elk daarvan heen loopt. Deze architectuur is gebaseerd op de verwerking van visuele informatie in de hersenen die via de ogen wordt vastgelegd door het netvlies. Dankzij deze diepte kan een netwerk complexe structuren leren, zonder dat hiervoor onrealistisch grote hoeveelheden gegevens nodig zijn.

Dan hebben we neuronen, cellichamen en signalen. Een biologisch of artificieel neuraal netwerk bestaat uit een groot aantal eenvoudige eenheden, neuronen, die signalen ontvangen en naar elkaar overbrengen. De neuronen zijn eenvoudige informatieverwerkers, die bestaan uit een cellichaam en draden die de neuronen met elkaar verbinden. Het grootste deel van de tijd doen zij niets anders dan afwachten of er via de draden signalen binnenkomen.

Het is belangrijk om te begrijpen dat één enkele neuron op zichzelf niet zo indrukwekkend is en maar een beperkte functionaliteit heeft. De kracht van een neuraal netwerk is dat er zeer veel neuronen aan elkaar gelinkt kunnen worden en daarmee het systeem erg complex wordt. Elk neuron reageert op een specifieke manier, die in de loop der tijd ook kan veranderen, op de binnenkomende signalen.

Neurale netwerken zijn ontworpen voor een verschillend aantal doelen. Beter inzicht in de werking van het menselijk brein, functies achterhalen, maar voor ons project is het vooral interessant dat neurale netwerken gebruikt kunnen worden voor een betere artificiële intelligentie en voor machine learning.

Ze hebben een aantal belangrijke eigenschappen namelijk:

-	Een neuraal netwerk bestaat uit een groot aantal neuronen dat elk op zich stukjes informatie verwerkt, in plaats van een CPU die alles alleen doet. 
-	Opslag en verwerking van gegevens zijn niet gescheiden zoals in een traditionele computer.

Hiervoor is parallelle verwerking nodig en dit kan gesimuleerd worden op een traditionele CPU, maar dit is niet zo performant. Er is specifieke hardware nodig die veel informatie tegelijkertijd kan verwerken. Gelukkig kunnen GPU’s dit en dit is ook de beste oplossing voor ons project. 

# Numpy

Numpy is een library voor wetenschappelijk rekenen in Python. De library maakt het mogelijk om multidimensionale array objecten aan te maken, snelle berekeningen voor arrays, arrays sorteren en manipuleren, lineaire algebra, etc. 

Numpy vectorizeert alle arrays. Dit betekent dat het mogelijk is om array a en array b te vermenigvuldigen zonder daarvoor een lus te gebruiken. (Gewoon a * b volstaat in plaats van for each element in a doe …) Dit bewaart de readability van de code en bovendien gebruikt Numpy in the achtergrond pre-compiled (zeer snelle) C++ code die de berekeningen doet. De loops worden door die C++ code in de achtergrond gebruikt.

Vectorizeren van arrays heeft als voordelen:

-	Gevectorizeerde code is korter en meer leesbaar
-	Minder lijnen code betekent minder bugs
-	De code lijkt meer op wiskundige formules dus het is makkelijker om wiskundige formules te nabootsen
-	De code is meer “pythonic” zonder inefficiënte, moeilijk te lezen for loops

# Tensorflow

Tensorflow is een open source machine learning framework/library ontwikkeld door het Google Brain team. De library bundelt een groot aantal machine learning en deep learning modellen en algoritmes en maakt ze bruikbaar in één geheel. De front-end en API’s zijn gemaakt in Python voor makkelijk gebruik, maar de back-end is gebouwd in C++ voor de performantie die we gewoon zijn van C++.

Tensorflow kan deep neural networks trainen en runnen voor bijvoorbeeld image recognition, natural language processing, etc.

De library werkt door de mogelijkheid aan te bieden om dataflow grafen aan te maken. Deze structuren beschrijven hoe data zich beweegt door een graaf. Elke node in de graaf stelt een wiskundige operatie voor en elke verbinding hiertussen is een multidimensionaal array of tensor.

Het is mogelijk om Tensorflow te gebruiken op een waaier van verschillende apparaten. Een locale machine, een cluster in de cloud, CPU’s of GPU’s, etc.

De kracht van Tensorflow ligt vooral in het abstraheren van de “technische” details en de focus ligt vooral op effectief developpen. De algoritmes en details worden door Tensorflow zelf afgehandelt en de programmeur kan zich focussen op het ontwerpen van de software en de logica erachter. 

Doordat Tensorflow ondersteund wordt door Google, verloopt de ontwikkeling van en rond de library zeer vlot. Zo is de ondersteuning en opzet van Tensorflow widespread ondersteund. 

Bronnen:

- https://www.tensorflow.org/
- https://www.infoworld.com/article/3278008/what-is-tensorflow-the-machine-learning-library-explained.html
- https://numpy.org/doc/1.20/user/whatisnumpy.html
- https://tweakers.net/reviews/5901/4/neurale-netwerken-de-beslissende-kracht-achter-internet-het-trainen-van-het-netwerk.html
- https://course.elementsofai.com/nl-be/5/1
- https://www.programmersought.com/article/76061559899/
- https://towardsdatascience.com/everything-you-ever-wanted-to-know-about-computer-vision-heres-a-look-why-it-s-so-awesome-e8a58dfb641e
- https://opencv.org/about/

