# Hướng dẫn 3 — Triển khai

**Yêu cầu tên file:** Viết script triển khai chính tên **`task3.py`** (hoặc `app.py` được chấp nhận) chứa ứng dụng Gradio. Script này phải:
- Nạp pipeline đã lưu (`best_pipeline.joblib`) và các file phụ trợ (`feature_order.json`, `target_mapping.json`) khi khởi động.
- Cung cấp hàm dự đoán trả về **(label, probabilities)**; probabilities phải trả về một dict mapping label → float.
- Giao diện Gradio phải có input controls tương ứng tất cả các thuộc tính đầu vào (Dropdown cho categorical, Number cho numeric).  
- Có ví dụ đầu vào (examples) để kiểm thử nhanh trong UI.

Triển khai `task3.py` lên **Hugging Face Spaces (Gradio)**, kèm theo `requirements.txt` (ghi phiên bản `gradio`, `scikit-learn`, `pandas`, `joblib`, v.v.) và `README.md` hướng dẫn nhanh.

- Triển khai pipeline tốt nhất (preprocessing + model) dưới dạng ứng dụng web dùng **Gradio**, và publish trên **Hugging Face Spaces**.

---

## 1. Lưu pipeline đã huấn luyện
- Lưu pipeline đầy đủ bằng `joblib` (bao gồm tiền xử lý):

```python
import joblib
joblib.dump(best_pipeline, 'best_pipeline.joblib')
```

---

## 2. Tạo ứng dụng Gradio
- Ứng dụng phải:
  - Nạp pipeline lúc khởi động.
  - Có input control cho tất cả thuộc tính (Dropdown cho categorical, Number cho numeric).
  - Khi submit, chạy `pipeline.predict` để trả về nhãn, và có thể trả xác suất bằng `pipeline.predict_proba`.

Ví dụ `app.py` (mang tính minh họa):
```python
import joblib
import gradio as gr
import pandas as pd

pipe = joblib.load('best_pipeline.joblib')
classes = pipe.named_steps['clf'].classes_

def predict_fn(gender, age, ...):
    # Build a single-row DataFrame with the same feature names and order
    row = pd.DataFrame([{ 'Gender': gender, 'Age': age, ... }])
    pred = pipe.predict(row)[0]
    prob = pipe.predict_proba(row)[0]
    return pred, {c: float(p) for c, p in zip(classes, prob)}

iface = gr.Interface(
    fn=predict_fn,
    inputs=[gr.Dropdown(...), gr.Number(...), ...],
    outputs=[gr.Label(num_top_classes=1), gr.JSON()],
    title='Obesity Level Prediction',
)

if __name__ == '__main__':
    iface.launch()
```

---

## 3. Đưa lên Hugging Face Spaces
1. Tạo repository mới (type: Gradio) trên Hugging Face.
2. Đưa các file sau vào repo:
   - `app.py`
   - `best_pipeline.joblib`
   - `requirements.txt` (ghi `gradio`, `scikit-learn`, `pandas`, `joblib`, ...)
3. Push repo; Spaces sẽ tự deploy ứng dụng.

---

## 4. Kiểm thử & Nộp bài
- Test ứng dụng với các input mẫu từ dataset để kiểm tra nhất quán.
- Nộp: URL Hugging Face Space, `app.py`, `best_pipeline.joblib`, `requirements.txt`, và README hướng dẫn nhanh.