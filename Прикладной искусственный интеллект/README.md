
# Линейная регрессия — пример приложения

## Что реализовано
- Загрузка Excel-файла
- Деление данных на train/test
- Обучение Linear Regression
- Верификация модели
- Сохранение модели
- Возможность добавлять новые данные

## Установка

```bash
pip install pandas scikit-learn openpyxl joblib
```

## Запуск обучения

```bash
python train_model.py
```

После запуска:
- обучится модель
- выведутся метрики
- создастся файл `model.pkl`

## Использование модели

```bash
python predict.py
```

## Как добавить новые данные
В `predict.py` можно добавлять новые строки:

```python
new_data = pd.DataFrame([
    {
        "feed_1": 45,
        "feed_2": 78,
        "feed_3": 61
    }
])
```
