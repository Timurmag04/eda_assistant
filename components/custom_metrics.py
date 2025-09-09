import streamlit as st
import pandas as pd

def compute_custom_metric(df, formula):
    """
    Вычисляет пользовательскую метрику на основе введенной формулы.
    Args:
        df: pandas DataFrame с данными.
        formula: строка с формулой (например, "df['age'] * 2").
    Returns:
        Результат вычисления или None в случае ошибки.
    """
    try:
        result = pd.eval(formula, local_dict={'df': df}, engine='python')
        return result
    except Exception as e:
        st.error(f"Ошибка в формуле: {e}")
        return None