import os
import numpy as np
import cv2
import pickle
import tensorflow as tf
from flask import Flask, request, render_template_string
from skimage import feature
import base64

app = Flask(__name__)

# --- 1. KONFIGURASI MODEL & SCALER ---
# Ganti dengan nama file model Keras (.h5) dan Scaler (.pkl) Anda
MODEL_PATH = 'ethnicity_ann_model.h5' 
SCALER_PATH = 'scaler_ann.pkl'

# TENTUKAN LABEL SESUAI URUTAN TRAINING (Alphabetical order biasanya)
\
CLASS_NAMES = ['asian', 'africa', 'eropa'] 

model = None
scaler = None

# Load System
try:
    # Load Model Keras
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load Scaler (Pickle)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
        
    print("✅ System (Keras Model + Scaler) Loaded Successfully")
except Exception as e:
    print(f"❌ Error loading system: {e}")


# --- 2. PIPELINE FUNCTIONS (Sesuai kode Anda) ---
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def detect_and_crop_face(image):
    if image is None: return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    if len(faces) > 0:
        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
        x, y, w, h = faces[0]
        return gray[y:y+h, x:x+w], (x, y, w, h)
    return None, None

def preprocess_image_pipeline(image, output_size=(64, 64)):
    if image is None: return None
    resized = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized)
    return equalized

def extract_lbp_features(image, P=8, R=1, method='uniform'):
    lbp = feature.local_binary_pattern(image, P, R, method=method)
    n_bins = 59 if method == 'uniform' else 256
    hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist

def extract_hog_features(image):
    hog_feats = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
                            cells_per_block=(2, 2), block_norm='L2-Hys',
                            visualize=False, feature_vector=True)
    return hog_feats

def fuse_features(lbp, hog):
    return np.concatenate([lbp, hog])


# --- 3. FLASK ROUTES ---

@app.route('/', methods=['GET'])
def home():
    return render_template_string("""
    <div style="text-align:center; padding:50px; font-family: sans-serif;">
        <h1>🌍 Race & Ethnicity Classifier</h1>
        <p>Powered by TensorFlow Keras + LBP + HOG</p>

        <form action="/predict" method="post" enctype="multipart/form-data">
            <input type="file" name="file" id="fileInput" required accept="image/*"><br><br>
            <img id="preview" src="#" style="display:none; width:300px; border-radius:8px; border:2px solid #333;"/><br><br>
            <button type="submit" style="padding:10px 20px; font-weight:bold;">🔎 Identifikasi</button>
        </form>
    </div>

    <script>
        const input = document.getElementById('fileInput');
        const preview = document.getElementById('preview');
        input.onchange = evt => {
            const [file] = input.files;
            if (file) {
                preview.src = URL.createObjectURL(file);
                preview.style.display = "block";
            }
        }
    </script>
    """)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        image_bytes = file.read()
        
        # Decode Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img_original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 1. Pipeline: Detect Face
        face_img, rect = detect_and_crop_face(img_original)

        if face_img is None:
            return "<h3>❌ Wajah tidak ditemukan. Gunakan gambar dengan wajah menghadap depan.</h3><br><a href='/'>Kembali</a>"

        # 2. Pipeline: Extraction
        processed = preprocess_image_pipeline(face_img)
        lbp_feat = extract_lbp_features(processed)
        hog_feat = extract_hog_features(processed)
        
        # 3. Fuse & Scale
        features = fuse_features(lbp_feat, hog_feat)
        features = features.reshape(1, -1) # Reshape untuk input model
        features = scaler.transform(features) # Normalize

        # 4. Keras Prediction
        preds = model.predict(features) # Output berupa probability array
        
        # Logika Prediksi (Multi-class)
        # Jika model anda Binary (output 1 neuron), gunakan logika if > 0.5
        # Jika Multi-class (output > 1 neuron + Softmax):
        class_idx = np.argmax(preds[0])
        label = CLASS_NAMES[class_idx] if class_idx < len(CLASS_NAMES) else f"Class {class_idx}"
        confidence = np.max(preds[0]) * 100

        # Draw Rectangle on Original Image
        x, y, w, h = rect
        cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Convert to Base64
        _, buffer = cv2.imencode('.jpg', img_original)
        encoded = base64.b64encode(buffer).decode('utf-8')

        return render_template_string(f"""
            <div style="text-align:center; padding:40px; font-family: sans-serif;">
                <h2>Hasil Prediksi</h2>
                <img src="data:image/jpeg;base64,{encoded}" style="width:350px; border-radius:10px;"/><br><br>
                
                <h3>Prediksi: <span style="color:blue;">{label}</span></h3>
                <p>Confidence: <b>{confidence:.2f}%</b></p>
                <br>
                <a href="/" style="font-size:18px;">🔙 Coba Lagi</a>
            </div>
        """)

    except Exception as e:
        return f"Error: {e}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)