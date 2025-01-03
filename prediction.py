from time import time
import torch
from torchvision.transforms import v2
from PIL import Image
from pathlib import Path
from onnxruntime import InferenceSession

# Globale Variablen
last_detected_time = 0


# Lädt das Model aus dem model-Ordner und gibt es zurück
# Wird einmalig beim Programmstart und "Load Model" ausgeführt
def load(model_dir: str):
    DRIVE_MODEL_NAME = "model_03c_Fahrmodell_Training.onnx"  # <- Hier muss der Name des Models stehen
    SIGN_MODEL_NAME = "Stoppschild_AI.onnx"   # <- Hier muss der Name des Models stehen

    drive_model_path = Path(model_dir) / DRIVE_MODEL_NAME
    sign_model_path = Path(model_dir) / SIGN_MODEL_NAME

    assert drive_model_path.exists(), f"Model does not exist: {drive_model_path}"
    assert sign_model_path.exists(), f"Model does not exist: {sign_model_path}"

    drive_model = InferenceSession(str(drive_model_path))
    sign_model = InferenceSession(str(sign_model_path))
    return drive_model, sign_model


# Diese Funktion bekommt das Kamerabild und das Model (welches in load() geladen wurde) und soll angle und speed zurückgeben
# Die Funktion wird während des Self-Driving-Modus in einer Schleife aufgerufen
def step(img, models) -> tuple[float, float]:  # angle, speed
    global last_detected_time

    drive_model, sign_model = models

    # Bild in Tensor umwandeln
    img_tensor = img_to_tensor(img)
    # Winkel vorhersagen
    angle_output = drive_model.run(None, {drive_model.get_inputs()[0].name: img_tensor.numpy()})
    angle = angle_output[0].item()

    # Sign vorhersagen
    img_tensor = img_to_tensor(img)
    sign_output = sign_model.run(None, {sign_model.get_inputs()[0].name: img_tensor.numpy()})
    sign_value = sign_output[0].item()

    speed = 30

    time_since_last_detection = time() - last_detected_time

    if time_since_last_detection <= 5:
        return angle, 0

    if time_since_last_detection <= 10:
        return angle, speed

    if sign_value > 0.8:
        last_detected_time = time()
        return angle, 0

    return angle, speed


# Hilfsfunktion um ein PIL-Image in ein Tensor umzuwandeln (ähnlich was im Dataset gemacht wird)
def img_to_tensor(image: Image) -> torch.Tensor:
    image = image.resize((160, 120))
    image = v2.functional.pil_to_tensor(image)
    image = image/255.0
    image = image.unsqueeze(0)
    return image

# Self test
if __name__ == '__main__':
    from PIL import Image

    model_path = "models"  # / "models"
    test_img = Image.new("RGB", (320, 240))
    model = load(model_path)
    angle, speed = step(test_img, model)
    print(f"Prediction: angle={angle}, speed={speed}")
