import pandas as pd
import numpy as np
from scipy import stats

def get_extended_stats(df, selected_cols=None):
    """Возвращает расширенную статистику для выбранных числовых столбцов."""
    if selected_cols is None:
        selected_cols = df.select_dtypes(include=['number']).columns
    stats_df = pd.DataFrame()
    for col in selected_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            stats_dict = {
                'Столбец': col,
                'Среднее': df[col].mean(),
                'Медиана': df[col].median(),
                'Минимум': df[col].min(),
                'Максимум': df[col].max(),
                'Стд. отклонение': df[col].std(),
                'Коэф. вариации (%)': (df[col].std() / df[col].mean() * 100) if df[col].mean() != 0 else np.nan,
                'Q1 (25%)': df[col].quantile(0.25),
                'Q3 (75%)': df[col].quantile(0.75),
                'Асимметрия': stats.skew(df[col]),
                'Эксцесс': stats.kurtosis(df[col])
            }
            stats_df = pd.concat([stats_df, pd.DataFrame([stats_dict])], ignore_index=True)
    return stats_df

def detect_outliers(df, selected_cols=None):
    """Обнаруживает аномалии в числовых столбцах с использованием метода IQR."""
    if selected_cols is None:
        selected_cols = df.select_dtypes(include=['number']).columns
    outliers = {}
    for col in selected_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers[col] = df[outlier_mask][col].index.tolist()
    return outliers

def get_correlations(df, method="pearson"):
    """Возвращает корреляционную матрицу для числовых столбцов."""
    numeric_df = df.select_dtypes(include=['number'])
    if len(numeric_df.columns) > 1:
        return numeric_df.corr(method=method)
    return None