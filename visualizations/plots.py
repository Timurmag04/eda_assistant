import streamlit as st
import plotly.express as px

def plot_histogram(df, column):
    """
    Строит гистограмму для указанной колонки.
    """
    if column in df.columns:
        fig = px.histogram(df, x=column, title=f"Гистограмма для {column}")
        st.plotly_chart(fig)
    else:
        st.error("Выбранная колонка не найдена.")


def plot_scatter(df, x_col, y_col):
    fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}")
    st.plotly_chart(fig)