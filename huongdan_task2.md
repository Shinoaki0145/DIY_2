# Hướng dẫn 2 — Đánh giá

- Các chỉ số bắt buộc: Accuracy, Ma trận nhầm lẫn (heatmap), Macro Precision, Macro Recall, Macro F1-score, Macro ROC–AUC (đa lớp, trung bình macro).

---

## 1. Tải dự đoán và nhãn
- Với mỗi mô hình đã lưu, cần có:
  - Nhãn dự đoán trên tập test (`y_pred`).
  - Xác suất dự đoán trên tập test (`y_proba`).

Có thể nạp lại pipeline đã lưu và chạy `pipe.predict` / `pipe.predict_proba`.

Ví dụ:
```python
import joblib
pipe = joblib.load('best_pipeline.joblib')
y_pred = pipe.predict(X_test)
y_proba = pipe.predict_proba(X_test)
```

---

## 2. Tổng quan hiệu năng
**Accuracy**
```python
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_test, y_pred)
print('Accuracy:', acc)
```

**Ma trận nhầm lẫn (hiển thị)**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names)
```
Lưu cả ma trận thô và ma trận chuẩn hóa để phân tích.

---

## 3. Báo cáo phân loại (trung bình Macro)
Tính Macro Precision, Macro Recall, Macro F1:
```python
from sklearn.metrics import precision_score, recall_score, f1_score
precision_macro = precision_score(y_test, y_pred, average='macro')
recall_macro = recall_score(y_test, y_pred, average='macro')
f1_macro = f1_score(y_test, y_pred, average='macro')
```
Ghi lại các chỉ số này cho từng mô hình và so sánh trong bảng.

---

## 4. Khả năng phân biệt mô hình — Macro ROC–AUC (đa lớp)
- Dùng `label_binarize` để chuyển y_test về ma trận nhị phân.
- Dùng `predict_proba` để lấy xác suất đầu ra.
- Tính AUC trung bình macro với `multi_class='ovr'`.

Ví dụ:
```python
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
n_classes = len(pipe.named_steps['clf'].classes_)
y_test_bin = label_binarize(y_test, classes=range(n_classes))
auc_macro = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
```

---

## 5. Phân tích & so sánh
- So sánh các mô hình theo các chỉ số trên.
- Chọn mô hình tốt nhất và giải thích dựa trên chỉ số và đặc điểm dữ liệu (ví dụ: mất cân bằng lớp, bằng chứng overfitting).
- Dùng ma trận nhầm lẫn để phân tích các cặp nhầm lẫn phổ biến và nêu giả thuyết (ví dụ: các lớp BMI liền kề dễ bị nhầm).

Nộp: một báo cáo/ô notebook ngắn tóm tắt so sánh và mô hình được chọn.

---

## 6. Ghi chú điểm số
Phần này chiếm **25%** điểm bài tập.

---

## 7. Kiểm tra & khả năng tái tạo
- Lưu `results_summary.csv` gồm tên mô hình và các chỉ số bắt buộc.
- Lưu ma trận nhầm lẫn dưới dạng PNG để đưa vào báo cáo.