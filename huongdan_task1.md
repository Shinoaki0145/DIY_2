# Mô hình hóa
## Dữ liệu yêu cầu
- Sử dụng `ObesityDataset.csv` (2111 bản ghi, 14 thuộc tính đầu vào + biến mục tiêu `NObesity`).
- Các lớp mục tiêu: Underweight, Normal, Overweight, Obesity I, Obesity II, Obesity III.

---

## 1. Chuẩn bị dữ liệu cho mô hình (BẮT BUỘC)
Thực hiện đúng các bước sau; *không được xử lý thủ công ngoài Pipeline*.

1.1 Chia Train/Test
- Chia dữ liệu thành tập huấn luyện và kiểm thử theo tỷ lệ **80/20**.
- Dùng `stratify=y` để giữ phân phối lớp.
- Khóa phép sinh ngẫu nhiên `random_state=42`.

Ví dụ:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
```

1.2 Tiền xử lý (bên trong Pipeline)
- **Mã hóa các thuộc tính phân loại bằng OneHotEncoder** (không dùng LabelEncoder cho các đặc trưng).
- **Chuẩn hóa các thuộc tính số** nếu cần (ví dụ: StandardScaler).
- Dùng `ColumnTransformer` và `Pipeline` để đảm bảo các bước tiền xử lý nằm trong pipeline mô hình.

Ví dụ:
```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
numeric_cols = ['age', ...]  # replace with numeric columns
cat_cols = ['gender', 'family history with overweight', ...]
preproc = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])
```

**Quan trọng:** KHÔNG thực hiện tiền xử lý thủ công bên ngoài pipeline (vi phạm sẽ làm mất yêu cầu bài tập).

---

## 2. Xây dựng mô hình phân loại
- Huấn luyện ít nhất **n + 1** mô hình phân loại đa lớp, với `n` là số thành viên trong nhóm. Ví dụ: nếu có 3 người → ít nhất 4 mô hình.
- Mỗi mô hình phải là **pipeline end-to-end**: `Pipeline([('preproc', preproc), ('clf', <classifier>)])`.
- Gợi ý mô hình: LogisticRegression, DecisionTreeClassifier, RandomForestClassifier, KNeighborsClassifier, GaussianNB, SVC (với `probability=True`), v.v.

Ví dụ huấn luyện:
```python
from sklearn.ensemble import RandomForestClassifier
pipe_rf = Pipeline([('preproc', preproc), ('clf', RandomForestClassifier(random_state=42))])
pipe_rf.fit(X_train, y_train)
```

---

## 3. Kết quả mô hình cần chuẩn bị cho phần Đánh giá
Với mỗi mô hình đã huấn luyện, tạo và lưu các kết quả sau trên **tập kiểm thử**:
- **Nhãn lớp dự đoán** (`y_pred`) — lưu vào CSV để nộp.
- **Xác suất dự đoán cho từng lớp** (`y_proba`) — cần cho ROC–AUC đa lớp (lưu CSV hoặc npy).

Ví dụ:
```python
y_pred = pipe_rf.predict(X_test)
y_proba = pipe_rf.predict_proba(X_test)
# Save
import pandas as pd
pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_csv('rf_predictions.csv', index=False)
pd.DataFrame(y_proba, columns=pipe_rf.named_steps['clf'].classes_).to_csv('rf_probas.csv', index=False)
```

---

## Sản phẩm nộp cho phần Mô hình hóa
- Các pipeline đã huấn luyện cho tất cả mô hình (ưu tiên lưu bằng `joblib` hoặc `pickle`).
- Các file dự đoán (nhãn + xác suất) cho mỗi mô hình trên tập test.
- Một ô notebook ngắn mô tả lựa chọn tiền xử lý và danh sách `n+1` mô hình đã huấn luyện.

---

## Ghi chú điểm số
Phần này chiếm **50%** điểm của bài tập (Triển khai thuật toán phân loại).