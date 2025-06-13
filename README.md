# klasifikasi-batik-mobilenetv2-finetune

# Domain Proyek  
Proyek Machine Learning ini berada pada domain _sosial, budaya,_ dan _industri kreatif_, khususnya pada pelestarian dan inovasi motif batik Indonesia. Sistem ini mengimplementasikan _dual-model AI_:  
1. Model klasifikasi (_MobileNet v2_) untuk mengenali, mengkategorikan, dan menganalisis makna serta asal-usul motif batik.  

## Latar Belakang  
Batik diakui sebagai warisan budaya dunia oleh UNESCO. Namun, pemahaman masyarakat tentang makna serta sejarah motif batik mengalami penurunan, dengan minimnya dokumentasi digital dan kurangnya eksplorasi motif baru yang tetap otentik. Karya seni AI terbaru (misal: DALL-E, Stable Diffusion) telah membuktikan potensi AI dalam domain kreatif, tapi belum ada sistem yang spesifik untuk motif batik Indonesia. Berdasarkan hasil wawancara dengan pengrajin dan komunitas budaya, terlihat urgensi solusi digital yang tetap menjaga nilai filosofi dan keaslian batik.  

## Business Understanding  

### Problem Statements  
1. Bagaimana membangun sistem _machine learning_ untuk mengklasifikasikan dan mengenali motif batik beserta unsur budaya yang dikandungnya?  
2. Bagaimana evaluasi dan validasi kualitas model AI berdasarkan akurasi dan _relevansi budaya_?  

### Goals  
1. Membangun _pipeline_ AI untuk klasifikasi motif batik (mengidentifikasi jenis, asal, dan makna motif).  
2. Mendokumentasikan motif-motif digital yang dianalisis demi pelestarian budaya batik secara modern.  

## Data Understanding  

Dataset yang digunakan dikumpulkan dari riset dosen dan Repositori Digital Batik Jawa Timur, terdiri dari:  
- **420 gambar batik terkurasi**, disertai label (4 tema: Aristektur, Budaya, Flora, Fauna).    

Data 420 gambar tersebut akan diproses melalui _augmentation_ untuk memperkaya variasi.  

### Variabel pada Dataset  
- **Gambar Batik (_image_)**: Foto beresolusi 640x640 piksel.  
- **Label Motif**: Kategori (Arsitektur, Budaya, Fauna, Flora).    

### Kualitas Data  
Sebagian dataset memiliki kualitas gambar terbatas. Untuk mengatasi ini, diterapkan:  
- _Data augmentation_   
- Batch preprocessing untuk normalisasi warna dan skala  

## Data Preparation  

Tahapan persiapan data yang dilakukan:   
- _One-hot encoding_ pada kolom label motif  
- _Augmentation_ (rotasi, zoom, flip, perubahan warna)  
- _Standardization_: Normalisasi piksel ke rentang [0,1]  
- _Split_ dataset jadi train/test (80/20) agar evaluasi obyektif   

## Modeling  

### Model Klasifikasi (_MobileNet v2_)  
Sistem klasifikasi dibangun berdasarkan _MobileNet v2_ custom.  
- **Arsitektur**: 2 convolutional block + dense layer.  
- **Format penyimpanan model**: tflite
- **Optimasi**: _Batch processing_, _learning rate schedule_  
Model di-train pada gambar batik untuk mengenali pola spesifik masing-masing label.  

#### Hasil Evaluasi 
- **Akurasi CNN pada data kelas-kelas motif mencapai:**  
  - Sekitar 90–92% pada akurasi, 74-78% val akurasi.  
- **Makna motif hasil generasi** dinilai cukup otentik oleh validator manual, walaupun ada beberapa kasus di mana motif AI perlu supervisi/koreksi budaya.  

## Penyelesaian Permasalahan  
Solusi model ini memungkinkan:  
1. **Klasifikasi otomatis motif batik** berdasarkan visual, mempercepat katalogisasi & dokumentasi batik daerah.   
2. **Validasi berlapis** baik dari segi teknis (metrik akurasi) maupun budaya (feedback pengrajin/dosen budaya).  
4. **Integrasi teknologi AI untuk pelestarian**, selaras dengan kebutuhan industri kreatif, pendidikan, dan promosi budaya digital.  

## Kesimpulan  
Proyek ini berhasil mendemonstrasikan kombinasi model CNN untuk klasifikasi motif dalam konteks warisan budaya. Hasil evaluasi menunjukkan performa teknis yang baik, serta relevansi budaya yang dijaga melalui validasi manual. Sistem ini dapat dikembangkan lebih lanjut menjadi platform web/AI service untuk edukasi dan industri kreatif, serta berpotensi memperkaya dokumentasi digital batik Indonesia secara signifikan.  
