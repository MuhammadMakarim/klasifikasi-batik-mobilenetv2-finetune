import streamlit as st  
import tensorflow as tf  
import numpy as np  
from PIL import Image  
import io, os, tempfile  
import gdown  

# --- PAGE CONFIG ---  
st.set_page_config(  
    page_title="Intelligent Batik AI Dashboard",  
    page_icon="🎨",  
    layout="wide",  
)  

# --- MODERN CSS, SOLID BG, JELAS SEMUA KONTEN ---  
st.markdown("""  
    <style>  
    body, .stApp {  
        background: #00000 !important;  
    }  
    .stTabs [data-baseweb="tab-list"] {  
        justify-content: center;  
        background: #ffffff;  
        box-shadow: 0 2px 12px rgba(60,60,60,0.13);  
        border-radius: 15px 15px 0 0;  
        margin-bottom: 2em;  
        font-weight: bold;  
    }  
    .stTabs [data-baseweb="tab"] {  
        padding: 1em 1.6em;  
        font-size: 1.15rem;  
        color: #1e2c4a !important;  
        font-weight: 600;  
    }  
    .block-container {  
        max-width: 950px;  
        margin: 3em auto 2em auto;  
        background: #000;  
        border-radius: 18px;  
        box-shadow: 0 8px 32px 0 rgba(63,69,87,0.12);  
        padding: 2.5em 2.4em 2em 2.4em;  
    }  
    h1, h2, h3 {  
        color: #16213E;  
        font-weight: 800;  
    }  
    .stAlert {  
        font-size: 1rem;  
        font-weight: 500;  
    }  
    </style>  
""", unsafe_allow_html=True)  

# --- LOAD TFLITE MODEL ---  
GDRIVE_MODEL_ID = "1uoZKDy1knCgxVO4gTn5hwuQ-11rkJ2hf"  
url        = f"https://drive.google.com/uc?id={GDRIVE_MODEL_ID}"  
tmp_dir    = tempfile.gettempdir()  
model_path = os.path.join(tmp_dir, f"{GDRIVE_MODEL_ID}.tflite")  

if not os.path.exists(model_path):  
    gdown.download(url, output=model_path, quiet=False)  

interpreter    = tf.lite.Interpreter(model_path=model_path)  
interpreter.allocate_tensors()  
input_details  = interpreter.get_input_details()  
output_details = interpreter.get_output_details()  
class_names    = ["Arsitektur", "Budaya", "Fauna", "Flora"]  

def do_predict(image: Image.Image):  
    img_array = np.array(image.convert("RGB").resize((224,224)))/255.0  
    interpreter.set_tensor(input_details[0]["index"], [img_array.astype(np.float32)])  
    interpreter.invoke()  
    probs = interpreter.get_tensor(output_details[0]["index"])[0]  
    idx   = np.argmax(probs)  
    return class_names[idx], float(probs[idx])  

# --- DASHBOARD CONTENT ---  
tabs = st.tabs([  
    "🏠 Beranda",  
    "🤖 Tentang Model",  
    "📁 Upload Gambar",  
    "📷 Inference Kamera"  
])  

# --- TAB 1: Beranda ---  
with tabs[0]:  
    st.markdown("## 🎨 Selamat Datang di Batik AI Dashboard!")  
    st.markdown("""  
Dashboard ini mendukung pelestarian serta inovasi batik melalui kecerdasan buatan.  
**Anda dapat:**  
- Mengenal inovasi AI yang diterapkan dalam bidang batik.  
- Mengunggah dan mengenali motif batik menggunakan model MobileNetV2.  
- Melakukan klasifikasi motif batik langsung melalui kamera.  
    """)  
    st.info("Proyek ini membantu dokumentasi otomatis dan kreasi motif baru untuk pengrajin, desainer, serta edukator batik di Indonesia.")  

# --- TAB 2: Tentang Model (penjelasan detail dan modern) ---  
with tabs[1]:  
    st.markdown("## 🤖 Penjelasan Model AI Batik yang Digunakan")  
    st.subheader("Latar Belakang dan Tujuan")  
    st.write("""  
Model AI Batik ini dikembangkan sebagai solusi permasalahan menurunnya pemahaman masyarakat terhadap makna, sejarah, dan keberagaman motif batik, sekaligus menstimulasi inovasi. Proyek ini memanfaatkan dua pendekatan utama:  
- **Model Computer Vision:** Untuk menganalisis, mengklasifikasikan, serta mengidentifikasi motif batik yang telah ada.  
    """)  
    st.subheader("Arsitektur dan Teknologi")  
    st.write("""  
Model **MobileNetV2** dipilih sebagai backbone klasifikasi motif batik karena kemampuannya memberikan keseimbangan antara efisiensi komputasi dan akurasi tinggi, sangat pas untuk kebutuhan deployment di sistem mobile/web maupun resource terbatas.  

**Pipeline pengembangan model:**  
- **Input:** Gambar batik resolusi 224×224 piksel, format RGB.  
- **Preprocessing:** Resize, normalization, augmentasi gambar.  
- **Training:** Dengan transfer learning menggunakan data batik Jawa Timur sebagai data utama, dan batik dari daerah lain sebagai data tambahan (berisi ribuan motif), pembelajaran fine-tuning dengan label terstruktur: `Arsitektur`, `Budaya`, `Fauna`, `Flora`.  
- **Inference & Deployment:** Model dikonversi ke TensorFlow Lite agar ringan dan responsif untuk aplikasi web/mobile Anda.  
    """)  

    st.success("""  
**Keunggulan & Dampak:**  
- Model sangat ringan, cepat, dan hemat resource sehingga ideal untuk digunakan masyarakat umum.  
- Tingkat akurasi tinggi pada motif batik autentik Indonesia.  
- Sistem AI ini dapat membantu pengrajin memperluas kreativitas tanpa melupakan makna filosofi batik.  
- Mendukung dokumentasi digital dan pelestarian kekayaan budaya lokal.  
    """)  

# --- TAB 3: Upload Gambar ---  
with tabs[2]:  
    st.markdown("## 📁 Upload Gambar Batik untuk Prediksi")  
    uf = st.file_uploader("Pilih file gambar batik (.jpg, .png)", type=["jpg","jpeg","png"])  
    if uf:  
        img = Image.open(io.BytesIO(uf.read()))  
        st.image(img, caption="Gambar yang Diunggah", use_container_width=True)  
        label, conf = do_predict(img)  
        st.success(f"Prediksi: **{label}** ({conf:.2%})")  

# --- TAB 4: Kamera Prediksi ---  
with tabs[3]:  
    st.markdown("## 📷 Prediksi Batik Melalui Kamera")  
    cam = st.camera_input("Ambil foto batik Anda")  
    if cam:  
        img = Image.open(io.BytesIO(cam.read()))  
        st.image(img, caption="Foto Kamera", use_container_width=True)  
        label, conf = do_predict(img)  
        st.success(f"Prediksi: **{label}** ({conf:.2%})")  