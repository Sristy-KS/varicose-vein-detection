from flask import Flask, render_template, request
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import DepthwiseConv2D
import numpy as np
import os

app = Flask(__name__)

# -----------------------------
# Paths (GitHub-safe relative paths)
# -----------------------------
YOLO_MODEL_PATH = "runs/train/best.pt"
SEVERITY_MODEL_PATH = "severity_model_gpu.h5"

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# -----------------------------
# Patch DepthwiseConv2D
# (Fix EfficientNet compatibility issue)
# -----------------------------
orig_init = DepthwiseConv2D.__init__

def patched_init(self, *args, **kwargs):
    kwargs.pop("groups", None)
    orig_init(self, *args, **kwargs)

DepthwiseConv2D.__init__ = patched_init


# -----------------------------
# Load Models
# -----------------------------
print("Loading YOLO model...")
yolo_model = YOLO(YOLO_MODEL_PATH)

print("Loading severity model...")
severity_model = load_model(SEVERITY_MODEL_PATH, compile=False)

print("All models loaded successfully.\n")

SEVERITY_LABELS = ["mild", "moderate", "severe"]


# -----------------------------
# Severity Prediction Function
# -----------------------------
def predict_severity(image_path):
    """Predict Mild or Moderate (convert Severe → Moderate)."""
    img = image.load_img(image_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    preds = severity_model.predict(x, verbose=0)
    pred_label = SEVERITY_LABELS[np.argmax(preds)]

    # Convert severe → moderate
    if pred_label.lower() == "severe":
        pred_label = "moderate"

    return pred_label.capitalize()


# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    message = None

    if request.method == "POST":
        file = request.files.get("file")
        if not file or file.filename == "":
            return render_template("index.html", result="Please upload an image.")

        # Save uploaded image
        img_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(img_path)

        # Run YOLO Detection
        device_choice = "cuda" if tf.config.list_physical_devices('GPU') else "cpu"
        results = yolo_model.predict(
            source=img_path,
            save=False,
            imgsz=640,
            conf=0.45,
            iou=0.45,
            show_labels=False,
            show_conf=False,
            device=device_choice,
            verbose=False
        )

        # Decision based on YOLO result
        if len(results[0].boxes) > 0:
            severity = predict_severity(img_path)
            message = f"Varicose veins detected — Severity: {severity}"
        else:
            message = "No varicose veins detected."

    return render_template("index.html", result=message)


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
