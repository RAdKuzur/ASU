import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_excel("питатели.xlsx", header=5)
df.columns = ["time", "feed_1", "feed_2", "feed_3", "target"]
df = df[["feed_1", "feed_2", "feed_3", "target"]]
df = df.replace(r"^\s*$", None, regex=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()
X = df[["feed_1", "feed_2", "feed_3"]]
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
print("MAE:", round(mae, 2))
print("R2 :", round(r2, 4))

joblib.dump(model, "model.pkl")
print("Модель сохранена")