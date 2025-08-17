<img width="760" height="605" alt="image" src="https://github.com/user-attachments/assets/c5c7551b-5918-410e-9601-6842209da772" />
AI-OST-Challenge in Rapperswil
Was wir machen mussten:
Eine kleine AI trainieren damit das Auto selbst fährt.

Auto soll geradeaus fahren und um Kurven lenken
Stoppschilder erkennen und anhalten
Nach dem Stopp wieder weiterfahren

Meine Lösung
Hab zwei Modelle trainiert:

model_03c_Fahrmodell_Training.onnx - für Lenkwinkel
Stoppschild_AI.onnx - für Stoppschilder

Das Auto bekommt Kamerabilder und gibt Lenkwinkel + Geschwindigkeit zurück.
Logik

Normales Fahren: Speed 30, Winkel vom Modell
Stoppschild erkannt (>0.8 confidence): 5 Sekunden stopp
Nach 10 Sekunden: wieder normal fahren


Link: https://www.ost.ch/de/die-ost/campus/campus-rapperswil-jona/nachwuchs/ai-challenge/ai-challenge-2024




