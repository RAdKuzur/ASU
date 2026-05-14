import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

#Загрузка данных

columns = [
    "Class",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
]

df = pd.read_csv("wine.data", header=None, names=columns)

#Анализ пропусков

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

print("\nПропущенные значения:")
print(missing)

# Визуализация пропусков
plt.figure(figsize=(10, 5))
missing_percent.plot(kind='bar')
plt.title("Доля пропущенных значений")
plt.ylabel("Процент")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#Анализ дисбаланса классов

plt.figure(figsize=(6, 4))
df["Class"].value_counts().sort_index().plot(kind='bar')

plt.title("Распределение классов")
plt.xlabel("Класс")
plt.ylabel("Количество")
plt.show()

print("\nКоличество объектов по классам:")
print(df["Class"].value_counts())

#Корреляция признаков

corr = df.corr(numeric_only=True)["Class"].abs().sort_values(ascending=False)

print("\nКорреляция признаков с целевой переменной:")
print(corr)

top_features = corr.index[1:11]

# Визуализация корреляции
plt.figure(figsize=(8, 5))
corr[top_features].sort_values().plot(kind='barh')

plt.title("Топ признаков по корреляции")
plt.xlabel("Корреляция")
plt.tight_layout()
plt.show()



X = df[top_features]
y = df["Class"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Логистическая регрессия

model = LogisticRegression(
    max_iter=5000,
    class_weight='balanced'
)

param_grid = {
    "C": [0.01, 0.1, 1, 10, 100]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='f1_weighted'
)

grid.fit(X_train_scaled, y_train)

best_model = grid.best_estimator_

print("\nЛучший параметр C:")
print(grid.best_params_)

#Предсказания

y_pred = best_model.predict(X_test_scaled)
y_proba = best_model.predict_proba(X_test_scaled)

#Метрики качества

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

auc = roc_auc_score(
    y_test,
    y_proba,
    multi_class='ovr'
)

print("\nМетрики модели:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"AUC-ROC  : {auc:.4f}")
#Матрица ошибок
cm = confusion_matrix(y_test, y_pred)

print("\nМатрица ошибок (Confusion Matrix):")
print(cm)

#Важность признаков

# Усредняем коэффициенты для multiclass
importance = np.mean(np.abs(best_model.coef_), axis=0)

importance_df = pd.DataFrame({
    "Feature": top_features,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\nВажность признаков:")
print(importance_df)

#График важности
plt.figure(figsize=(8, 5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()

plt.title("Наиболее значимые признаки")
plt.xlabel("Важность")

plt.tight_layout()
plt.show()

#Интерпретация ошибок

print("""
Интерпретация ошибок:

Ложноотрицательные ошибки (False Negative)
обычно критичнее для производства,
так как модель пропускает дефектный объект.

Ложноположительные ошибки (False Positive)
менее опасны, потому что приводят только
к дополнительной проверке объекта.
""")


new_samples = pd.DataFrame([
    {
        "Flavanoids": 3.5,
        "OD280/OD315 of diluted wines": 3.2,
        "Total phenols": 2.8,
        "Proline": 1100,
        "Hue": 1.05,
        "Alcalinity of ash": 15.5,
        "Proanthocyanins": 2.0,
        "Nonflavanoid phenols": 0.25,
        "Malic acid": 1.8,
        "Alcohol": 13.8
    },
    {
        "Flavanoids": 1.2,
        "OD280/OD315 of diluted wines": 1.5,
        "Total phenols": 1.6,
        "Proline": 500,
        "Hue": 0.8,
        "Alcalinity of ash": 21.0,
        "Proanthocyanins": 1.1,
        "Nonflavanoid phenols": 0.45,
        "Malic acid": 3.5,
        "Alcohol": 12.1
    },
    {
        "Flavanoids": 0.8,
        "OD280/OD315 of diluted wines": 1.0,
        "Total phenols": 1.2,
        "Proline": 400,
        "Hue": 0.65,
        "Alcalinity of ash": 24.0,
        "Proanthocyanins": 0.9,
        "Nonflavanoid phenols": 0.5,
        "Malic acid": 4.2,
        "Alcohol": 11.8
    }
])

new_samples_scaled = scaler.transform(new_samples)
predicted_classes = best_model.predict(new_samples_scaled)

print("\nПредсказание для новых тестовых данных:")

for i, pred in enumerate(predicted_classes):
    print(f"Объект {i + 1} --> Класс: {pred}")