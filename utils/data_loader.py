import pandas as pd

def load_data(file):
    """
    Загружает данные из CSV файла и проверяет наличие пропусков.
    Возвращает DataFrame и словарь с информацией о пропусках.
    """
    try:
        df = pd.read_csv(file)
        missing_info = {
            "total_missing": df.isnull().sum().sum(),
            "numeric_cols": df.select_dtypes(include=['number']).columns.tolist(),
            "categorical_cols": df.select_dtypes(include=['object', 'category']).columns.tolist(),
            "missing_per_col": df.isnull().sum().to_dict()
        }
        return df, missing_info
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return None, None