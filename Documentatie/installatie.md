# Installatiegids

## Requirements

Voor de installatie van het project gaan we er van uit dat er toegang is tot:
* een Nvidia Jetson Xavier NX Dev Kit.
* een 1080p camera of beter die op _minstens_ 4 meter hoogte gemonteerd staat en dit onder een hoek van rond de 25 graden.

## Jetson opstarten

Om het project op te zetten, maken we gebruik van een Jetpack image (deze is gebaseerd op Ubuntu versie 18.04). De guide om de laatste Jetpack versie (in ons geval versie 4.5) op de Jetson te installeren staat beschreven op [de website van Nvidia](https://developer.nvidia.com/embedded/learn/get-started-jetson-xavier-nx-devkit). 

## Dependencies

Eenmaal de Jetson opgestart is kunnen de dependencies voor het project installeren. Hiervoor maken we gebruik van een requirements tekst bestand.

Clone de repository van [ons project op GitHub](https://github.com/lucacelea/Afstudeerproject/).

```bash
git clone https://github.com/lucacelea/Afstudeerproject/

cd Afstudeerproject
```

Installeer pip3 en upgrade.

```bash
sudo apt install python3-pip
sudo python3 -m pip install --upgrade pip
```

Installeer de dependencies vanuit het requirements bestand.

```bash
sudo pip install -r requirements.txt
```



