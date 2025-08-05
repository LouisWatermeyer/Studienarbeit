# Studienarbeit Louis Watermeyer

Dieses Repository stellt den Code für die Studienarbeit "Objekterkennung von Würfeln und deren Wert mittels OpenCV"

## Einstellungen
Es können mehrere Einstellungen in der Datei settings.py vorgenommen werden:
 - DEBUG_DRAWINGS: True | False
   - True: Es werden zusätzliche Bilder ausgegeben, die den Prozess an verschiedenen Stellen visualisieren
   - False: Es wird nur das Endergebnis gezeigt
 - IS_WINDOWS: True | False
   - Das Betriebssystem muss angegeben werden. Zu False ändern, wenn Linux genutzt wird
 - WINDOWS_CAMERA_INDEX: int
   - Der Index der Kamera die unter Windows genutzt werden soll
 - LINUX_CAMERA_PATH
   - Der Pfad zum Kameradevice das unter Linux genutzt werden soll

## Programm starten
Um das Programm zu starten, müssen einmalig die Abhängigkeiten isntalliert werden
```bash
pip install opencv-python
pip install numpy
pip install scikit-learn
pip install scipy
```

Um das Programm zu starten, muss die Datei dice-recognition.py mit Python ausgeführt werden.
```bash
python dice-recognition.py
```


