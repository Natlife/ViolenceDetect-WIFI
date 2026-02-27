# ViolenceDetectByWIFI

> **BullyDetect: Detecting School Physical Bullying with Wi-Fi and Deep Wavelet Transformer**

---

## 📁 Cấu trúc Project

```
WiFi-BullyDetect/
├── config.py                  ← Tất cả cài đặt tập trung tại đây
├── dataset.py                 ← DataLoader + augmentation
├── trainer.py                 ← Training loop (early stopping, scheduler, logging)
├── evaluator.py               ← Test evaluation + plots
├── train.py                   ← Entry point: training
├── test.py                    ← Entry point: testing / inference
├── requirements.txt
├── WiFi_BullyDetect_Colab.ipynb  ← Google Colab notebook
├── model/
│   ├── __init__.py
│   └── dwt_transformer.py     ← Deep Wavelet Transformer model
├── scripts/
│   └── prepare_dataset.py     ← Download + preprocess dataset
└── data/
    ├── raw/                   ← Dữ liệu thô từ Kaggle
    └── preprocessed/
        ├── train/  (X.npy, y.npy)
        ├── val/    (X.npy, y.npy)
        └── test/   (X.npy, y.npy)
```

---

## ⚡ Quick Start (Windows / Linux)

### 1. Cài đặt môi trường

```bash
# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

# Cài dependencies
pip install -r requirements.txt
```

### 2. Tải dataset từ Kaggle

**Bước 1:** Lấy Kaggle API key

- Vào [kaggle.com](https://kaggle.com) → Account → Create API Token
- Tải về file `kaggle.json`
- **Windows:** đặt vào `C:\Users\<tên_user>\.kaggle\kaggle.json`
- **Linux:** đặt vào `~/.kaggle/kaggle.json` rồi chạy `chmod 600 ~/.kaggle/kaggle.json`

**Bước 2:** Tải và xử lý dataset

```bash
pip install kaggle
python scripts/prepare_dataset.py --kaggle
```

Hoặc nếu bạn **đã tải thủ công** và giải nén vào `data/raw/`:

```bash
python scripts/prepare_dataset.py
```

> **Cấu trúc thư mục data/raw/ mong đợi:**
>
> ```
> data/raw/
>   normal/       ← các file .npy chứa CSI khi không có bắt nạt
>   bullying/     ← các file .npy chứa CSI khi có bắt nạt
> ```
>
> Hoặc: `data/raw/<label>_<id>.npy` (label = 0 hoặc 1)

### 3. Train

```bash
# Train với cài đặt mặc định
python train.py

# Train với tùy chỉnh
python train.py --run_name my_experiment --epochs 100 --lr 0.001 --batch 32

# Train baseline CNN (để so sánh)
python train.py --model CNN --run_name baseline_cnn
```

### 4. Test / Evaluate

```bash
# Đánh giá trên test set
python test.py --run_name my_experiment

# Predict một file CSI đơn lẻ
python test.py --run_name my_experiment --input path/to/csi_sample.npy
```

---

## ☁️ Google Colab

Mở file `WiFi_BullyDetect_Colab.ipynb` trên Colab, chạy từng cell theo thứ tự.

**Tóm tắt các bước trong notebook:**

1. Mount Google Drive
2. Upload / copy project vào Colab
3. Cài dependencies
4. Upload `kaggle.json` để tải dataset
5. Chạy `prepare_dataset.py`
6. Chạy training
7. Xem kết quả, lưu về Drive

---

## ⚙️ Cấu hình (config.py)

Tất cả hyperparameter và đường dẫn nằm trong `config.py`. Các thông số quan trọng:

| Tham số           | Mặc định | Mô tả                                       |
| ----------------- | -------- | ------------------------------------------- |
| `MODEL_NAME`      | `"DWT"`  | Model: `"DWT"`, `"CNN"`                     |
| `NUM_SUBCARRIERS` | `30`     | Số subcarrier WiFi                          |
| `WINDOW_SIZE`     | `500`    | Độ dài time-window                          |
| `EMBED_DIM`       | `64`     | Embedding dimension                         |
| `NUM_HEADS`       | `4`      | Số attention heads                          |
| `NUM_LAYERS`      | `3`      | Số Transformer layers                       |
| `WAVELET`         | `"db4"`  | Loại wavelet (pywt)                         |
| `WAVELET_LEVEL`   | `3`      | Số cấp DWT                                  |
| `BATCH_SIZE`      | `32`     | Batch size                                  |
| `NUM_EPOCHS`      | `100`    | Số epochs                                   |
| `LEARNING_RATE`   | `1e-3`   | Learning rate                               |
| `EARLY_STOP`      | `15`     | Early stopping patience                     |
| `NUM_WORKERS`     | `0`      | DataLoader workers (**Windows: phải là 0**) |

---

## 📊 Kết quả

Sau khi train xong, kết quả được lưu tại `results/`:

```
results/
├── weights/<run_name>/
│   ├── best.pth     ← checkpoint tốt nhất
│   └── last.pth     ← checkpoint cuối
├── plots/<run_name>/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── training_history.png
│   └── test_report.txt
└── logs/<run_name>/
    ├── training_log.csv
    └── tb/          ← TensorBoard logs (nếu cài)
```

**Xem TensorBoard (nếu đã cài):**

```bash
pip install tensorboard
tensorboard --logdir results/logs
```

---

## 🔧 Troubleshooting

**Lỗi `RuntimeError: DataLoader worker` trên Windows:**

```python
# config.py
NUM_WORKERS = 0  # bắt buộc 0 trên Windows
```

**Lỗi `ModuleNotFoundError: pywt`:**

```bash
pip install PyWavelets
```

**Lỗi `No data found`:**

- Kiểm tra cấu trúc thư mục `data/raw/` theo hướng dẫn ở trên
- Chạy lại `python scripts/prepare_dataset.py`

**Train quá chậm (không có GPU):**

- Giảm `WINDOW_SIZE` xuống 250
- Giảm `BATCH_SIZE` xuống 16
- Giảm `NUM_EPOCHS` xuống 30
- Hoặc dùng Colab với GPU miễn phí

---

## 📄 Citation

```bibtex
@ARTICLE{
  author={Nat(Ndluong)},
  title={Violence Detect: Notifying from violence by WIFI},
}
```
