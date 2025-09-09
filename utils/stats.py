import pandas as pd
import streamlit as st

def get_basic_stats(df):
    """
    Возвращает описательную статистику DataFrame.
    """
    return df.describe()

def get_correlations(df, method="pearson"):
    """
    Возвращает корреляционную матрицу.
    """
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    if numeric_df.empty:
        st.warning("Нет числовых колонок для расчета корреляции.")
        return None
    return numeric_df.corr(method=method)