import base64
import importlib.util
import io
import os
import cv2
import numpy as np
import safetensors.torch
import torch
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request
from PIL import Image
from transformers import pipeline

app = Flask(__name__)

# Direktori dan path model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "arcface_r50.safetensors")
IRESNET_SOURCE_PATH = os.path.join(BASE_DIR, "py-feat", "feat", "identity_detectors", "arcface", "iresnet.py")

# Device configuration (CPU atau CUDA jika tersedia)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Detektor wajah Haar Cascade dari OpenCV untuk auto-crop wajah
frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")

def load_arcface_model():
    """Memuat arsitektur IResNet-50 dan bobot safetensors secara standalone."""
    print(f"[*] Memuat model ArcFace dari {IRESNET_SOURCE_PATH}...")
    spec = importlib.util.spec_from_file_location("iresnet", IRESNET_SOURCE_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    
    net = mod.iresnet50()
    print(f"[*] Memuat bobot safetensors dari {MODEL_WEIGHTS_PATH}...")
    state_dict = safetensors.torch.load_file(MODEL_WEIGHTS_PATH)
    net.load_state_dict(state_dict, strict=False)
    net.to(device)
    net.eval()
    print("[*] Model ArcFace R50 berhasil dimuat dan siap digunakan!")
    return net

# Inisialisasi ArcFace
model = load_arcface_model()

# Inisialisasi Model Klasifikasi Emosi (FER)
print("[*] Memuat model deteksi emosi wajah (FER)...")
emotion_pipeline = pipeline(
    "image-classification",
    model="dima806/facial_emotions_image_detection",
    device=-1
)
print("[*] Model deteksi emosi berhasil dimuat dan siap digunakan!")

# Database in-memory sederhana untuk fitur 1:N
face_database = []

# Kamus pemetaan emosi ke bahasa Indonesia, emoji, dan warna
EMOTION_META = {
    "happy": {"id": "Senang / Bahagia", "emoji": "😃", "color": "text-emerald-400", "bg": "bg-emerald-500", "badge": "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"},
    "neutral": {"id": "Netral / Tenang", "emoji": "😐", "color": "text-slate-300", "bg": "bg-slate-400", "badge": "bg-slate-500/20 text-slate-300 border-slate-500/30"},
    "surprise": {"id": "Kaget / Terkejut", "emoji": "😲", "color": "text-amber-400", "bg": "bg-amber-500", "badge": "bg-amber-500/20 text-amber-300 border-amber-500/30"},
    "angry": {"id": "Marah / Kesal", "emoji": "😠", "color": "text-rose-400", "bg": "bg-rose-500", "badge": "bg-rose-500/20 text-rose-300 border-rose-500/30"},
    "sad": {"id": "Sedih / Murung", "emoji": "😢", "color": "text-blue-400", "bg": "bg-blue-500", "badge": "bg-blue-500/20 text-blue-300 border-blue-500/30"},
    "fear": {"id": "Takut / Cemas", "emoji": "😨", "color": "text-purple-400", "bg": "bg-purple-500", "badge": "bg-purple-500/20 text-purple-300 border-purple-500/30"},
    "disgust": {"id": "Jijik / Muak", "emoji": "🤢", "color": "text-lime-400", "bg": "bg-lime-500", "badge": "bg-lime-500/20 text-lime-300 border-lime-500/30"},
}

def decode_base64_image(data_url):
    """Mendekode gambar base64 data URL menjadi objek PIL Image RGB."""
    if "," in data_url:
        data_url = data_url.split(",", 1)[1]
    image_bytes = base64.b64decode(data_url)
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return img

def pil_to_base64_thumbnail(pil_img, size=(112, 112)):
    """Mengubah PIL image menjadi base64 thumbnail untuk preview di browser."""
    thumb = pil_img.copy()
    thumb.thumbnail(size)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

def detect_and_crop_face(pil_img):
    """
    Mendeteksi wajah dan memotong (crop) hanya area wajah dengan sedikit margin.
    Jika wajah tidak terdeteksi otomatis, melakukan center-crop sebagai fallback.
    """
    np_img = np.array(pil_img)
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)

    # Deteksi wajah frontal
    faces = frontal_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=4, minSize=(35, 35)
    )

    # Jika tidak ditemukan, coba deteksi wajah profil
    if len(faces) == 0:
        faces = profile_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(35, 35)
        )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        mw = int(0.15 * w)
        mh = int(0.15 * h)
        x1 = max(0, x - mw)
        y1 = max(0, y - mh)
        x2 = min(np_img.shape[1], x + w + mw)
        y2 = min(np_img.shape[0], y + h + mh)
        cropped = pil_img.crop((x1, y1, x2, y2))
        return cropped, True, [int(x1), int(y1), int(x2), int(y2)]

    w, h = pil_img.size
    min_dim = min(w, h)
    left = (w - min_dim) // 2
    top = (h - min_dim) // 2
    cropped = pil_img.crop((left, top, left + min_dim, top + min_dim))
    return cropped, False, [int(left), int(top), int(left + min_dim), int(top + min_dim)]

def extract_face_embedding(pil_img):
    """Ekstraksi representasi vektor 512 dimensi ArcFace dari gambar."""
    cropped_face, face_detected, bbox = detect_and_crop_face(pil_img)
    img_resized = cropped_face.resize((112, 112), Image.Resampling.BILINEAR)
    np_img = np.array(img_resized, dtype=np.float32)

    tensor = torch.tensor(np_img).permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = (tensor * 2.0 - 1.0).to(device)

    with torch.no_grad():
        raw_emb = model(tensor)
        normed_emb = F.normalize(raw_emb, p=2, dim=1)

    return raw_emb.cpu().squeeze(0), normed_emb.cpu().squeeze(0), cropped_face, face_detected

def predict_emotions(pil_crop):
    """Memprediksi emosi pada potongan wajah (happy, angry, sad, dll.)."""
    raw_preds = emotion_pipeline(pil_crop)
    scores_dict = {item["label"].lower(): float(item["score"]) for item in raw_preds}

    # Buat list lengkap 7 emosi
    all_emotions = []
    for k, meta in EMOTION_META.items():
        score = scores_dict.get(k, 0.0)
        all_emotions.append({
            "key": k,
            "label_id": meta["id"],
            "emoji": meta["emoji"],
            "score": score,
            "percent": round(score * 100, 1),
            "color": meta["color"],
            "bg": meta["bg"],
            "badge": meta["badge"]
        })

    # Urutkan dari probabilitas tertinggi
    all_emotions.sort(key=lambda x: x["score"], reverse=True)
    dominant = all_emotions[0]
    return dominant, all_emotions

def calculate_match_confidence(cosine_sim, threshold=0.40):
    """Mengonversi nilai Cosine Similarity mentah menjadi persentase keyakinan."""
    if cosine_sim <= 0.0:
        return 0.0
    elif cosine_sim < threshold:
        return (cosine_sim / threshold) * 49.0
    else:
        return 50.0 + ((cosine_sim - threshold) / (1.0 - threshold)) * 50.0

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/status", methods=["GET"])
def api_status():
    return jsonify({
        "status": "ready",
        "identity_model": "ArcFace IResNet-50 (FP32)",
        "weights": "arcface_r50.safetensors",
        "emotion_model": "Facial Emotion ViT (FER-2013)",
        "face_detector": "OpenCV Haar Cascade + Smart Crop",
        "device": str(device),
        "embedding_dim": 512,
        "database_count": len(face_database)
    })

@app.route("/api/verify", methods=["POST"])
def api_verify():
    """Verifikasi 1:1 antara dua foto wajah dan analisis ekspresinya."""
    data = request.get_json(silent=True) or {}
    img_a_raw = data.get("image_a")
    img_b_raw = data.get("image_b")
    threshold = float(data.get("threshold", 0.40))

    if not img_a_raw or not img_b_raw:
        return jsonify({"error": "Dibutuhkan 2 gambar untuk verifikasi (image_a dan image_b)."}), 400

    try:
        pil_a = decode_base64_image(img_a_raw)
        pil_b = decode_base64_image(img_b_raw)

        _, emb_a, crop_a, detected_a = extract_face_embedding(pil_a)
        _, emb_b, crop_b, detected_b = extract_face_embedding(pil_b)

        # Prediksi ekspresi masing-masing foto
        dom_a, _ = predict_emotions(crop_a)
        dom_b, _ = predict_emotions(crop_b)

        # Hitung Cosine Similarity & Jarak Euclidean
        cosine_sim = F.cosine_similarity(emb_a.unsqueeze(0), emb_b.unsqueeze(0)).item()
        euclidean_dist = torch.dist(emb_a, emb_b).item()

        is_match = cosine_sim >= threshold
        confidence_percent = calculate_match_confidence(cosine_sim, threshold=threshold)

        return jsonify({
            "is_match": bool(is_match),
            "cosine_similarity": float(cosine_sim),
            "euclidean_distance": float(euclidean_dist),
            "confidence_percent": float(confidence_percent),
            "threshold": threshold,
            "decision": "MATCH" if is_match else "NO MATCH",
            "detected_a": detected_a,
            "detected_b": detected_b,
            "emotion_a": dom_a,
            "emotion_b": dom_b,
            "crop_a_thumb": pil_to_base64_thumbnail(crop_a),
            "crop_b_thumb": pil_to_base64_thumbnail(crop_b)
        })
    except Exception as e:
        return jsonify({"error": f"Gagal memproses gambar: {str(e)}"}), 500

@app.route("/api/emotion", methods=["POST"])
def api_emotion():
    """Deteksi 7 ekspresi emosi wajah pada satu gambar."""
    data = request.get_json(silent=True) or {}
    img_raw = data.get("image")

    if not img_raw:
        return jsonify({"error": "Gambar harus diisi."}), 400

    try:
        pil_img = decode_base64_image(img_raw)
        cropped_face, detected, bbox = detect_and_crop_face(pil_img)
        dominant, all_emotions = predict_emotions(cropped_face)

        return jsonify({
            "success": True,
            "dominant": dominant,
            "emotions": all_emotions,
            "face_detected": detected,
            "crop_thumb": pil_to_base64_thumbnail(cropped_face, size=(180, 180))
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/database/register", methods=["POST"])
def api_db_register():
    """Daftarkan wajah baru ke database lokal dengan auto-crop."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    img_raw = data.get("image")

    if not name or not img_raw:
        return jsonify({"error": "Nama dan gambar wajah harus diisi."}), 400

    try:
        pil_img = decode_base64_image(img_raw)
        raw_emb, normed_emb, crop_face, detected = extract_face_embedding(pil_img)
        thumb_b64 = pil_to_base64_thumbnail(crop_face, size=(80, 80))

        new_entry = {
            "id": len(face_database) + 1,
            "name": name,
            "thumbnail": thumb_b64,
            "embedding": normed_emb,
            "detected": detected
        }
        face_database.append(new_entry)

        return jsonify({"success": True, "id": new_entry["id"], "name": name, "total": len(face_database)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/database/list", methods=["GET"])
def api_db_list():
    """Mendapatkan daftar wajah yang terdaftar di database."""
    items = [
        {"id": item["id"], "name": item["name"], "thumbnail": item["thumbnail"]}
        for item in face_database
    ]
    return jsonify(items)

@app.route("/api/database/clear", methods=["POST"])
def api_db_clear():
    """Mengosongkan database."""
    face_database.clear()
    return jsonify({"success": True, "total": 0})

@app.route("/api/database/identify", methods=["POST"])
def api_db_identify():
    """Mencari identitas wajah yang paling mirip dari database (1:N)."""
    data = request.get_json(silent=True) or {}
    img_raw = data.get("image")
    threshold = float(data.get("threshold", 0.40))

    if not img_raw:
        return jsonify({"error": "Gambar uji harus diisi."}), 400

    if not face_database:
        return jsonify({"matches": []})

    try:
        pil_img = decode_base64_image(img_raw)
        _, query_emb, query_crop, detected = extract_face_embedding(pil_img)

        matches = []
        for entry in face_database:
            sim = F.cosine_similarity(query_emb.unsqueeze(0), entry["embedding"].unsqueeze(0)).item()
            conf = calculate_match_confidence(sim, threshold=threshold)
            matches.append({
                "id": entry["id"],
                "name": entry["name"],
                "thumbnail": entry["thumbnail"],
                "similarity": float(sim),
                "confidence_percent": float(conf),
                "is_match": bool(sim >= threshold)
            })

        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return jsonify({
            "matches": matches,
            "query_thumb": pil_to_base64_thumbnail(query_crop),
            "face_detected": detected
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/inspect", methods=["POST"])
def api_inspect():
    """Ekstraksi detail vektor 512 dimensi untuk analisis visual."""
    data = request.get_json(silent=True) or {}
    img_raw = data.get("image")

    if not img_raw:
        return jsonify({"error": "Gambar harus diisi."}), 400

    try:
        pil_img = decode_base64_image(img_raw)
        raw_emb, normed_emb, crop_face, detected = extract_face_embedding(pil_img)

        emb_list = raw_emb.tolist()
        return jsonify({
            "dimension": len(emb_list),
            "norm": float(torch.norm(raw_emb).item()),
            "mean": float(raw_emb.mean().item()),
            "std": float(raw_emb.std().item()),
            "min": float(raw_emb.min().item()),
            "max": float(raw_emb.max().item()),
            "face_detected": detected,
            "crop_thumb": pil_to_base64_thumbnail(crop_face),
            "vector_sample": emb_list[:32]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("\n=================================================================")
    print("  Face Recognition & Emotion Studio Aktif!")
    print("  - Identitas: ArcFace IResNet-50 (FP32)")
    print("  - Ekspresi : Facial Emotion ViT (7 Emosi Dasar)")
    print("  Buka browser di: http://localhost:4444")
    print("=================================================================\n")
    app.run(host="0.0.0.0", port=4444, debug=False)
