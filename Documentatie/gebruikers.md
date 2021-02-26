# Gebruikers Handleiding

### Opstarten van het systeem

Wanneer je alle dependencies hebt geïnstalleerd en de camera juist hebt opgehangen (zie installatie handleiding),  kan je beginnen met opstarten van het systeem.

Open de terminal en navigeer naar de directory waarin de launch.py file staat.
(Hoogstwaarschijnlijk onder Afstudeerproject/people-counting-opencv, als je onze github hebt gecloned)

Eens je in de directory zit, run je het volgende commando:

* OPENBLAS_CORETYPE=ARMV8 python3 launch.py

### Configuratie scherm

Eerste scherm na runnen van bovenstaande commando

![](C:\Users\jefqu\Desktop\scherm 1.JPG)

#### Legende:

* Video of camera: 
  Het systeem kan twee soorten inputs aan. Of je kiest voor camera, of je kiest een video bestand dat je lokaal op de computer hebt staan

* Model: 

  * Het systeem heeft twee modellen. Als je Alternatieve model uitschakelt dan sta je op het normale model

  * Skipped frames: hoe lager je dit getal zet, hoe meer frames het systeem gaat analyseren. Het systeem zal wel nauwkeuriger zijn, maar zal in ruil trager werken

    (Men kan het getal beter niet kleiner zetten dan 1)

* Logging:
  (een session is het starten van het systeem)

  Als je de seperate session logging files niet aanduid dan worden alle gegevens van over de verschillende sessions in één csv file gezet. Als je deze wel selecteert, dan krijg je een csv file per session.

* Save:
  Sla alles op in de config file

* Reset:
  Alles terug naar default waardes

* Run:
  Systeem runnen met geselecteerde configuraties

### Setup scherm

![](C:\Users\jefqu\Desktop\scherm 2.JPG)

Het setup scherm dient om de disinfectie zone aan te duiden. Hier zullen we het rode bord als dispenser gebruiken. Het is de bedoeling dat je cirkel vlak voor de dispenser trekt, op de plaats waar iemand zou staan. Met de muis selecteer je het middelpunt van de cirkel en sleep je de muis om de grote van de cirkel  te kiezen. Vervolgens krijg je dan onderstaand beeld te zien. 

![](C:\Users\jefqu\Desktop\scherm 3.JPG)

De cirkel dient dus om de disinfectie zone aan te duiden. Als een gedetecteerd persoon in deze cirkel staat voor een voldoende aantal seconden, dan zal deze geteld worden als een persoon die zijn handen heeft ontsmet. Wanneer je de cirkel opnieuw wilt trekken, druk je op de r-toets. Als je tevreden bent met de cirkel en de cirkel wilt bevestigen dan druk je op de c-toets.

### Running scherm

Vervolgens krijgen we het onderstaande scherm te zien.

![](C:\Users\jefqu\Desktop\scherm 4.JPG)

Dit is de live feed/video dat word afgespeeld. Zoals je ziet staat de door jouw geselecteerde cirkel op het scherm. Links zie je de telers. Disinfection is het aantal mensen dat hun handen hebben ontsmet tijdens de gestarte sessie. People zijn het totaal aantal mensen die gedetecteerd zijn tijdens de gestarte sessie. De gele lijn is de zone waarrond we de gedetecteerde mensen gaan tellen.
Als we het systeem willen afsluiten, dan drukken we op de q-toets.

### Data

Eens het systeem is afgesloten, kan je de sd-kaart uit de hardware halen en in uw computer steken. Vervolgens vind je de data in dezelfde directory als de launch.py file. In deze directory zie je één of meerdere csv files staan. Kies de juiste en open deze in Excel. Mogelijks krijg je onderstaand scherm te zien.

![](C:\Users\jefqu\Desktop\scherm 5.JPG)

Om de data in een meer overzichtelijke manier te bekijken, klik je onder Gegevens op importeer gegevens uit tekstbestand/CSV-bestand. (Links vanboven, rode cirkel op de foto)

![](C:\Users\jefqu\Desktop\scherm 6.JPG.png)

Als je de csv file op deze manier inlaad, krijg je het volgend beeld te zien.

![](C:\Users\jefqu\Desktop\scherm 7.JPG.png)

De verkregen data bestaat uit drie kolommen. De eerste kolom is de SESSION_ID, dit is de afgeronde epoch time van de opstarting van het systeem. De tweede kolom zijn de epoch times wanneer er een event heeft plaatsgevonden. De laatste kolom is de event kolom, deze kolom kan twee waardes bevatten: Entered (wanneer er een persoon geteld word bij het binnenkomen) en Disinfected (wanneer een persoon geteld word bij het ontsmetten van zijn handen).
