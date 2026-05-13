
import joblib
import pandas as pd
model = joblib.load("model.pkl")

new_data = pd.DataFrame([
    {
        "feed_1": 59,
        "feed_2": 74,
        "feed_3": 55
    }
])

prediction = model.predict(new_data)

print("Прогноз:", round(prediction[0], 2))
