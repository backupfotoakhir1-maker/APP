from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import os
import traceback

app = Flask(__name__)
CORS(app)

# ================== LOAD MODEL ==================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "MODEL_JAMUR_1.keras")

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# ================== LABEL ==================
label_map = ["edible", "poisonous"]
status_map = {
    "edible": "Aman Dimakan",
    "poisonous": "Beracun"
}

# ================== FUNGSI CEK KUALITAS GAMBAR ==================

def is_blur(img, threshold=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score < threshold


def is_dark(img, threshold=40):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return mean < threshold


# ================== TEST API ==================
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "API Jamur aktif",
        "endpoint": "/predict (POST, form-data: file)"
    })


# ================== PREDICT ==================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "file" not in request.files:
            return jsonify({"error": "File tidak ditemukan"}), 400

        file = request.files["file"]

        if file.filename == "":
            return jsonify({"error": "Nama file kosong"}), 400

        # ================== BACA GAMBAR ==================
        img_pil = Image.open(file.stream).convert("RGB")
        img_np = np.array(img_pil)
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ================== VALIDASI KUALITAS ==================
        if is_blur(img_cv):
            return jsonify({
                "error": "Foto terlalu blur. Silakan ambil ulang dengan fokus yang jelas."
            }), 400

        if is_dark(img_cv):
            return jsonify({
                "error": "Foto terlalu gelap. Silakan ambil ulang dengan pencahayaan cukup."
            }), 400

        # ================== PREPROCESS MODEL ==================
        img_resized = cv2.resize(img_cv, (224, 224))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_array = preprocess_input(img_rgb.astype(np.float32))
        img_array = np.expand_dims(img_array, axis=0)

        # ================== PREDIKSI ==================
        preds = model.predict(img_array)
        pred_index = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))

        # ================== CEK BUKAN JAMUR ==================
        if confidence < 0.60:
            return jsonify({
                "error": "Objek tidak dikenali sebagai jamur. Silakan foto jamur kembali."
            }), 400

        pred_label = label_map[pred_index]
        status = status_map[pred_label]

        # ================== HASIL ==================
        return jsonify({
            "jenis_jamur": pred_label,
            "status": status,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print("ERROR:", str(e))
        traceback.print_exc()
        return jsonify({"error": "Server error", "detail": str(e)}), 500


# ================== RUN SERVER ==================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
