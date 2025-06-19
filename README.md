Penjelasan Singkat Proyek: Sistem Deteksi Kualitas Roti Tawar
Proyek ini adalah sebuah sistem cerdas yang dirancang untuk mendeteksi kualitas roti tawar secara otomatis menggunakan citra digital. Tujuan utamanya adalah untuk mengidentifikasi apakah roti tawar dalam kondisi segar atau sudah berjamur.

Bagaimana Cara Kerjanya?
Sistem ini bekerja melalui serangkaian tahapan pemrosesan citra, yang didukung oleh teknologi Deep Learning terkini:

Pemuatan dan Peningkatan Kualitas Citra Awal:

Pengguna memilih gambar roti tawar yang ingin dianalisis.
Gambar tersebut kemudian melewati proses peningkatan kontras (Contrast Stretching). Ini dilakukan untuk membuat detail-detail pada roti, termasuk potensi jamur, menjadi lebih jelas dan mudah dianalisis di tahapan selanjutnya.
Segmentasi dengan HSV Thresholding dan Morfologi:

Gambar yang sudah ditingkatkan kontrasnya akan melalui proses HSV Thresholding. Ini adalah teknik untuk memisahkan area tertentu dalam gambar (misalnya, area yang mencurigakan seperti jamur) berdasarkan rentang warna Hue, Saturation, dan Value-nya. Hasilnya adalah sebuah "masker" biner yang menyoroti area potensial.
Selanjutnya, operasi morfologi citra (seperti Erosi, Dilasi, Opening, atau Closing) dapat diterapkan pada masker ini. Operasi ini membantu membersihkan noise, menyempurnakan bentuk area yang terdeteksi, atau menutup celah-celah kecil, sehingga segmentasi menjadi lebih akurat.
Deteksi Objek dan Klasifikasi Kualitas Roti dengan YOLOv8:

Ini adalah bagian inti dari proyek kita. Sistem menggunakan model YOLOv8 (You Only Look Once versi 8), sebuah algoritma deep learning yang sangat efisien dan akurat untuk deteksi objek.
Model YOLOv8 ini telah dilatih khusus untuk mengenali objek "roti" dan "jamur" pada citra roti tawar.
Ketika gambar roti dimasukkan ke model YOLOv8, ia akan secara otomatis mengidentifikasi dan menandai (dengan bounding box) di mana objek roti dan terutama jamur berada.
Berdasarkan hasil deteksi YOLOv8, sistem kemudian mengklasifikasikan kualitas roti:
Jika tidak ada jamur terdeteksi dengan tingkat kepercayaan yang memadai, roti akan diklasifikasikan sebagai "SEGAR".
Jika jamur terdeteksi, roti akan diklasifikasikan sebagai "BERJAMUR".
Manfaat Proyek Ini:
Proyek ini memberikan solusi otomatis untuk:

Membantu konsumen atau produsen dalam menentukan kualitas roti tawar secara cepat dan objektif.
Potensi untuk diintegrasikan dalam sistem kontrol kualitas di industri roti.
Mengurangi ketergantungan pada inspeksi visual manual yang bisa jadi subjektif.
