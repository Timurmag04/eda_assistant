import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.stats import get_basic_stats, get_correlations
from utils.filters import filter_data
from visualizations.plots import plot_histogram
from components.custom_metrics import compute_custom_metric

# Настройка страницы
st.set_page_config(page_title="EDA Assistant", layout="wide")

# Заголовок
st.title("EDA Assistant")

# Загрузка файла
uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])

if uploaded_file is not None:
    # Загружаем данные
    df = load_data(uploaded_file)
    if df is not None:
        st.session_state['df'] = df

        # Sidebar для навигации
        st.sidebar.header("Инструменты анализа")
        analysis_option = st.sidebar.selectbox(
            "Выберите действие",
            ["Предпросмотр данных", "Статистика", "Корреляции", "Гистограмма", "Кастомные метрики"]
        )

        # Фильтры в sidebar
        st.sidebar.subheader("Фильтры")
        filter_column = st.sidebar.selectbox("Выберите колонку для фильтра", df.columns)
        
        if filter_column in df.select_dtypes(include=['float64', 'int64']).columns:
            min_val = float(df[filter_column].min())
            max_val = float(df[filter_column].max())
            value_range = st.sidebar.slider(
                f"Диапазон для {filter_column}",
                min_value=min_val,
                max_value=max_val,
                value=(min_val, max_val)
            )
            df = filter_data(df, filter_column, value_range=value_range)
        elif filter_column in df.select_dtypes(include=['object']).columns:
            category = st.sidebar.selectbox(
                f"Категория для {filter_column}",
                df[filter_column].unique()
            )
            df = filter_data(df, filter_column, category=category)
        
        st.session_state['df'] = df  # Обновляем отфильтрованные данные

        # Основной контент
        if analysis_option == "Предпросмотр данных":
            st.subheader("Предпросмотр данных")
            st.dataframe(df.head(10))

        elif analysis_option == "Статистика":
            st.subheader("Описательная статистика")
            st.write(get_basic_stats(df))

        elif analysis_option == "Корреляции":
            st.subheader("Корреляционная матрица")
            corr = get_correlations(df, method="pearson")
            if corr is not None:
                st.write(corr)

        elif analysis_option == "Гистограмма":
            st.subheader("Построение гистограммы")
            column = st.selectbox("Выберите колонку", df.columns)
            plot_histogram(df, column)

        elif analysis_option == "Кастомные метрики":
            st.subheader("Кастомные метрики")
            formula = st.text_area(
                "Введите формулу (например, df['age'] * 2)",
                placeholder="Пример: df['age'] * 2 или df['salary'] / 1000"
            )
            if st.button("Вычислить"):
                if formula:
                    result = compute_custom_metric(df, formula)
                    if result is not None:
                        st.write("Результат вычисления:")
                        st.write(result)
                else:
                    st.warning("Введите формулу для вычисления.")

        # Кнопка для скачивания отфильтрованных данных
        st.download_button(
            label="Скачать отфильтрованные данные",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="filtered_data.csv",
            mime="text/csv"
        )
else:
    st.info("Пожалуйста, загрузите CSV-файл, чтобы начать анализ.")