control + F: final code
PHẦN 0 – HIỂU DATASET (BẮT BUỘC TRƯỚC KHI CODE)

📄 File: ObesityDataset.csv

2111 dòng, 14 features, 1 target

Target: NObesity (6 lớp → bài toán multi-class classification)

Nhóm thuộc tính
Loại	Cột
Numeric	age, FCVC, NCP, CH20, FAF, TUE
Categorical	gender, family_history, FAVC, CAEC, SMOKE, SCC, CALC, MTRANS

📌 Điểm giảng viên hay bắt lỗi
❌ Encode trước khi split
❌ Scale ngoài pipeline
❌ Không dùng ColumnTransformer

PHẦN 1 – MODELING (50%)
BƯỚC 1.1 – Train / Test Split (BẮT BUỘC ĐÚNG THAM SỐ)
• Test size = 20%
• Stratify = y
• random_state = 42


🎯 Mục đích:

Giữ phân bố 6 lớp obesity giống nhau ở train & test

📌 Bạn cần:

X = dataframe.drop("NObesity")

y = dataframe["NObesity"]

BƯỚC 1.2 – XÁC ĐỊNH CỘT NUMERIC & CATEGORICAL

Không đoán, phải liệt kê rõ trong code

numeric_features = [...]
categorical_features = [...]


📌 Giảng viên chấm rất kỹ phần này

BƯỚC 1.3 – PREPROCESSING (RẤT QUAN TRỌNG)
Yêu cầu đề bài

One-Hot Encoding cho categorical

Scaling cho numeric

TẤT CẢ nằm trong Pipeline

Cấu trúc chuẩn (bắt buộc nhớ):
ColumnTransformer
 ├── numeric_pipeline (StandardScaler)
 └── categorical_pipeline (OneHotEncoder)


📌 TUYỆT ĐỐI KHÔNG:

pd.get_dummies() bên ngoài

scaler.fit_transform() trước pipeline

BƯỚC 1.4 – XÂY DỰNG CÁC MÔ HÌNH
Số lượng mô hình

Ít nhất n + 1 model
(n = số thành viên nhóm)

Ví dụ nhóm 3 người → ≥ 4 models

Mỗi model phải là:
Pipeline(
  preprocessing
  → classifier
)

Model được khuyến nghị

Bạn có thể chọn:

Logistic Regression

Decision Tree

Random Forest

KNN

Naive Bayes

📌 Mỗi model:

fit(X_train, y_train)

predict(X_test)

predict_proba(X_test) ← bắt buộc cho ROC-AUC

PHẦN 2 – EVALUATION (25%)
BƯỚC 2.1 – METRICS BẮT BUỘC
1️⃣ Performance Overview

Accuracy

Confusion Matrix (vẽ heatmap)

📌 Giải thích:

Class nào bị nhầm nhiều?

Obesity I ↔ Obesity II có bị nhầm không?

2️⃣ Classification Report (MACRO)

BẠN PHẢI LẤY:

Macro Precision

Macro Recall

Macro F1-score

📌 Tại sao dùng macro?
→ Dataset multi-class + có thể imbalance

3️⃣ ROC – AUC (KHÓ NHẤT)

📌 Yêu cầu:

Macro-averaged ROC-AUC

Dùng predict_proba

Binarize label (OneVsRest)

🎯 Ý nghĩa:

Khả năng phân biệt tổng thể của model

BƯỚC 2.2 – SO SÁNH & PHÂN TÍCH

Bạn cần viết phân tích bằng lời, KHÔNG chỉ bảng số:

Gợi ý cấu trúc:

Model nào accuracy cao nhất

Model nào ổn định nhất (macro-F1)

Phân tích confusion matrix

Có imbalance không?

Vì sao model A > model B?

📌 Đây là phần ăn điểm 25%

PHẦN 3 – DEPLOYMENT (25%)
BƯỚC 3.1 – CHỌN MODEL TỐT NHẤT

👉 Dựa trên:

Accuracy

Macro-F1

ROC-AUC

Độ ổn định

📌 Không phải cứ RandomForest là tốt nhất → phải có lý do

BƯỚC 3.2 – LƯU & LOAD PIPELINE

Lưu TOÀN BỘ pipeline

Không chỉ model

Dùng joblib hoặc pickle

📌 Vì:

Input web → preprocessing → model → output

BƯỚC 3.3 – TẠO WEB BẰNG GRADIO

Web gồm:

Input cho 14 features

Button Predict

Output: NObesity

📌 Giao diện giống hình demo trang 5 của đề 

DIY 2

BƯỚC 3.4 – DEPLOY HUGGING FACE

Repo HuggingFace Spaces

SDK: Gradio

File chính: app.py

Upload model đã save

CHIẾN LƯỢC LÀM BÀI TRONG 3 GIỜ
Thời gian	Việc
30’	Đọc đề + phân tích dataset
60’	Modeling (pipelines + models)
40’	Evaluation
30’	So sánh & chọn model
20’	Gradio demo
20’	Kiểm tra & hoàn thiện







1️⃣ PSEUDO-CODE TỪNG PHẦN (ĐỂ BẠN TỰ CODE)
1. Đọc dữ liệu & tách X, y
LOAD ObesityDataset.csv

X = dataframe bỏ cột NObesity
y = dataframe["NObesity"]


📌 Lý do:

Tách rõ feature và target

Tránh leakage khi preprocessing

2. Train–Test Split (BẮT BUỘC)
SPLIT X, y thành:
- 80% train
- 20% test
- stratify = y
- random_state = 42


📌 Giải thích để ghi vào báo cáo

Việc sử dụng stratified split giúp đảm bảo phân bố các lớp béo phì được giữ nguyên giữa tập huấn luyện và kiểm tra, đặc biệt quan trọng trong bài toán phân loại nhiều lớp.

3. Phân loại cột (RẤT QUAN TRỌNG)
numeric_features = [
  age, FCVC, NCP, CH20, FAF, TUE
]

categorical_features = [
  gender, family_history_with_overweight,
  FAVC, CAEC, SMOKE, SCC,
  CALC, MTRANS
]


📌 Giảng viên hay hỏi miệng:

“Tại sao FCVC là numeric?” → Vì nó là tần suất dạng số.

4. Preprocessing Pipeline (KHÔNG ĐƯỢC LÀM NGOÀI)
Numeric pipeline
numeric_pipeline:
  StandardScaler

Categorical pipeline
categorical_pipeline:
  OneHotEncoder (handle_unknown = ignore)

ColumnTransformer
preprocessor:
  apply numeric_pipeline cho numeric_features
  apply categorical_pipeline cho categorical_features


📌 Câu thần chú

Không preprocessing ngoài pipeline → tránh data leakage

5. Xây dựng các mô hình (n + 1 models)

Ví dụ mỗi model đều có dạng:

pipeline_model_X:
  preprocessor
  → classifier_X


Ví dụ classifier:

Logistic Regression

Decision Tree

Random Forest

KNN

Naive Bayes

📌 Bắt buộc:

.fit(X_train, y_train)

.predict(X_test)

.predict_proba(X_test)

6. Lưu output để Evaluation
For mỗi model:
  y_pred = predict(X_test)
  y_proba = predict_proba(X_test)


📌 Lưu lại để dùng cho:

Confusion Matrix

Classification Report

ROC–AUC

2️⃣ TEMPLATE BÁO CÁO EVALUATION & ANALYSIS (COPY DÙNG ĐƯỢC)
2.1 Evaluation Metrics
Accuracy

Accuracy measures the overall proportion of correctly classified samples across all obesity classes.

Confusion Matrix

The confusion matrix provides insights into how the model misclassifies between obesity levels.
We observe that misclassifications mainly occur between neighboring classes such as Overweight and Obesity I, which is reasonable due to similar BMI ranges.

Classification Report (Macro)

Macro-averaged precision, recall, and F1-score are used to treat all classes equally, regardless of their sample size.

ROC–AUC (Macro)

Macro ROC–AUC evaluates the overall discriminative ability of the model across all classes using a one-vs-rest strategy.

2.2 Model Comparison
Model	Accuracy	Macro F1	Macro ROC–AUC
Logistic Regression			
Decision Tree			
Random Forest			
KNN			
2.3 Analysis & Discussion (PHẦN ĂN ĐIỂM)

Among all evaluated models, Random Forest achieved the best overall performance in terms of accuracy and macro F1-score.
The confusion matrix shows fewer misclassifications between distant obesity classes compared to other models.

Logistic Regression performed reasonably well but struggled to separate higher obesity levels, likely due to its linear decision boundary.

KNN showed sensitivity to feature scaling and may be affected by the high dimensionality after one-hot encoding.

📌 Kết luận mẫu

Based on the evaluation results and dataset characteristics, Random Forest is selected as the best-performing model for deployment.

3️⃣ GIẢI THÍCH ROC–AUC MULTI-CLASS (CỰC DỄ HIỂU)
3.1 ROC–AUC là gì (1 câu)

ROC–AUC đo khả năng mô hình phân biệt đúng giữa các lớp, không phụ thuộc vào threshold.

3.2 Vấn đề: nhiều hơn 2 lớp thì sao?

Dataset có 6 lớp → không thể vẽ 1 đường ROC duy nhất.

➡️ Giải pháp: One-vs-Rest (OvR)

3.3 One-vs-Rest là gì?

Ví dụ lớp Obesity I:

Xem Obesity I là Positive

5 lớp còn lại là Negative
→ tính ROC–AUC

Làm như vậy 6 lần → lấy trung bình (macro)

3.4 Macro ROC–AUC có ý nghĩa gì?

Macro ROC–AUC reflects the model’s average discriminative performance across all obesity levels, treating each class equally.

📌 Tại sao không dùng micro?
→ Vì class imbalance → macro công bằng hơn

4️⃣ HƯỚNG DẪN DEPLOY HUGGINGFACE (TỪNG BƯỚC)
BƯỚC 1 – Save pipeline
SAVE best_pipeline as "model.pkl"


📌 Không save riêng model
→ phải save cả preprocessing + model

BƯỚC 2 – Tạo file app.py

Pseudo-structure:

LOAD model.pkl

DEFINE function predict(
  gender, age, family_history, ...
):
  CREATE dataframe từ input
  prediction = pipeline.predict(data)
  RETURN prediction

BƯỚC 3 – Gradio Interface
gr.Interface(
  fn = predict,
  inputs = [
    Dropdown, Number, Slider, ...
  ],
  outputs = Text
)


📌 Input phải đủ 14 features

BƯỚC 4 – HuggingFace Spaces

Tạo account HF

New Space

SDK: Gradio

Upload:

app.py

model.pkl

requirements.txt

BƯỚC 5 – Test online

Nhập dữ liệu

Nhấn Predict

Trả về: NObesity




// Final Code
PHẦN 0 – IMPORT THƯ VIỆN
import pandas as pd
import numpy as np

# Train-test split
from sklearn.model_selection import train_test_split

# Preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score
)

# ROC-AUC multi-class
from sklearn.preprocessing import label_binarize

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Save model
import joblib

PHẦN 1 – LOAD DATASET
df = pd.read_csv("ObesityDataset.csv")

print(df.shape)
print(df.head())

PHẦN 2 – TÁCH X, y
X = df.drop(columns=["NObesity"])
y = df["NObesity"]

PHẦN 3 – TRAIN / TEST SPLIT (BẮT BUỘC ĐÚNG)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

PHẦN 4 – XÁC ĐỊNH CỘT NUMERIC & CATEGORICAL
numeric_features = [
    "Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"
]

categorical_features = [
    "Gender",
    "family_history_with_overweight",
    "FAVC",
    "CAEC",
    "SMOKE",
    "SCC",
    "CALC",
    "MTRANS"
]


📌 Tên cột phải đúng 100% với CSV
(Nếu khác → print(df.columns) để kiểm tra)

PHẦN 5 – PREPROCESSING PIPELINE (RẤT QUAN TRỌNG)
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

PHẦN 6 – XÂY DỰNG CÁC MÔ HÌNH (n + 1)
6.1 Logistic Regression
logistic_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

6.2 Decision Tree
dt_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=42))
])

6.3 Random Forest
rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ))
])

6.4 KNN
knn_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

6.5 Naive Bayes (lưu ý riêng)

⚠️ GaussianNB không làm việc trực tiếp với sparse matrix, nên cần mẹo nhỏ:

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

dense_transformer = FunctionTransformer(
    lambda x: x.toarray(), accept_sparse=True
)

nb_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("to_dense", dense_transformer),
    ("classifier", GaussianNB())
])

PHẦN 7 – TRAIN MODELS
models = {
    "Logistic Regression": logistic_pipeline,
    "Decision Tree": dt_pipeline,
    "Random Forest": rf_pipeline,
    "KNN": knn_pipeline,
    "Naive Bayes": nb_pipeline
}

for name, model in models.items():
    model.fit(X_train, y_train)

PHẦN 8 – EVALUATION
8.1 Accuracy + Confusion Matrix
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy:", acc)

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

8.2 Classification Report (MACRO)
for name, model in models.items():
    y_pred = model.predict(X_test)

    print(f"\n{name}")
    print(classification_report(y_test, y_pred))


📌 Lấy macro avg để ghi báo cáo

8.3 ROC–AUC MULTI-CLASS (KHÓ NHẤT)
classes = y.unique()
y_test_bin = label_binarize(y_test, classes=classes)

for name, model in models.items():
    y_proba = model.predict_proba(X_test)

    roc_auc = roc_auc_score(
        y_test_bin,
        y_proba,
        average="macro",
        multi_class="ovr"
    )

    print(f"{name} - Macro ROC-AUC: {roc_auc:.4f}")

PHẦN 9 – CHỌN & SAVE MODEL TỐT NHẤT

Giả sử Random Forest tốt nhất:

joblib.dump(rf_pipeline, "best_model.pkl")

PHẦN 10 – GRADIO APP (app.py)
import gradio as gr
import joblib
import pandas as pd

model = joblib.load("best_model.pkl")

def predict_obesity(
    Gender, Age, family_history, FAVC, FCVC, NCP,
    CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS
):
    data = pd.DataFrame([{
        "Gender": Gender,
        "Age": Age,
        "family_history_with_overweight": family_history,
        "FAVC": FAVC,
        "FCVC": FCVC,
        "NCP": NCP,
        "CAEC": CAEC,
        "SMOKE": SMOKE,
        "CH2O": CH2O,
        "SCC": SCC,
        "FAF": FAF,
        "TUE": TUE,
        "CALC": CALC,
        "MTRANS": MTRANS
    }])

    prediction = model.predict(data)[0]
    return prediction

Interface
interface = gr.Interface(
    fn=predict_obesity,
    inputs=[
        gr.Dropdown(["Male", "Female"]),
        gr.Number(),
        gr.Dropdown(["yes", "no"]),
        gr.Dropdown(["yes", "no"]),
        gr.Slider(1, 3),
        gr.Slider(1, 4),
        gr.Dropdown(["no", "Sometimes", "Frequently", "Always"]),
        gr.Dropdown(["yes", "no"]),
        gr.Slider(1, 3),
        gr.Dropdown(["yes", "no"]),
        gr.Slider(0, 3),
        gr.Slider(0, 2),
        gr.Dropdown(["no", "Sometimes", "Frequently", "Always"]),
        gr.Dropdown(["Automobile", "Bike", "Motorbike", "Public_Transportation", "Walking"])
    ],
    outputs="text",
    title="Obesity Level Prediction"
)

interface.launch()