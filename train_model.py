import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Tạo dữ liệu synthetic giống competition (bạn thay bằng train.csv thực tế sau)
np.random.seed(42)
n = 10000
data = {
    'ma_sv': range(1, n+1),
    'hoc_ky': np.random.randint(1, 9, n),
    'gpa': np.random.uniform(0.5, 4.0, n).round(2),
    'tin_chi_dat': np.random.randint(0, 25, n),
    'tin_chi_dk': np.random.randint(12, 30, n),
    'so_mon': np.random.randint(4, 10, n),
    'so_mon_fail': np.random.randint(0, 5, n),
    'ty_le_tham_gia': np.random.uniform(60, 100, n).round(1),
}
df = pd.DataFrame(data)

# Target: Cảnh báo học vụ (1 = có cảnh báo, dựa trên quy tắc thực tế competition)
df['canh_bao'] = ((df['tin_chi_dat'] / df['tin_chi_dk'] < 0.8) | 
                  (df['gpa'] < 2.0) | 
                  (df['so_mon_fail'] > 2)).astype(int)

# Features
X = df[['hoc_ky', 'gpa', 'tin_chi_dat', 'tin_chi_dk', 'so_mon', 'so_mon_fail', 'ty_le_tham_gia']]
y = df['canh_bao']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
model.fit(X_train, y_train)

# Đánh giá
pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Serialize thành pickle
with open('academic_warning_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model đã lưu: academic_warning_model.pkl")