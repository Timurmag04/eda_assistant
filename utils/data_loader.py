import pandas as pd
import streamlit as st

def load_data(file):
    """
    Загружает CSV файл и возвращает DataFrame.
    """
    try:
        df = pd.read_csv(file)
        return df
    except Exception as e:
        st.error(f"Ошибка при загрузке файла: {e}")
        return None