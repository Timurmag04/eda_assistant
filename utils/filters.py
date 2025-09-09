import streamlit as st
import pandas as pd

def filter_data(df, column, value_range=None, category=None):
    """
    Фильтрует DataFrame по колонке.
    value_range: кортеж (min, max) для числовых колонок.
    category: значение для категориальной колонки.
    """
    filtered_df = df.copy()
    if value_range and column in df.select_dtypes(include=['float64', 'int64']).columns:
        min_val, max_val = value_range
        filtered_df = filtered_df[(filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)]
    elif category and column in df.select_dtypes(include=['object']).columns:
        filtered_df = filtered_df[filtered_df[column] == category]
    return filtered_df