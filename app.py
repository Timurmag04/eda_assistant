import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.stats import get_basic_stats, get_correlations
from utils.filters import filter_data
from visualizations.plots import plot_histogram
from components.custom_metrics import compute_custom_metric

# Настройка страницы
st.set_page_config(page_title="EDA Assistant", layout="wide")

st.title("EDA Assistant")

# --- Инициализация состояния ---
if "df" not in st.session_state:
    st.session_state['df'] = None
if "original_df" not in st.session_state:
    st.session_state['original_df'] = None
if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.current_step = -1
if "filters_applied" not in st.session_state:
    st.session_state.filters_applied = False

# --- Функции истории изменений ---
def save_state(df):
    st.session_state.history = st.session_state.history[:st.session_state.current_step + 1]
    st.session_state.history.append(df.copy())
    st.session_state.current_step += 1
    st.session_state['df'] = df

def undo_action():
    if st.session_state.current_step > 0:
        st.session_state.current_step -= 1
        st.session_state['df'] = st.session_state.history[st.session_state.current_step]
        st.success("Действие отменено.")
        st.rerun()  # Перерисовываем интерфейс после отмены
    else:
        st.warning("Нет действий для отмены.")

def apply_filters_and_sort(base_df, filters, sort_config):
    filtered_df = base_df.copy()
    
    # Применение фильтров
    for col, config in filters.items():
        if col in filtered_df.columns:
            if pd.api.types.is_numeric_dtype(filtered_df[col]):
                # Числовой фильтр: границы min/max
                min_val = config.get('min')
                max_val = config.get('max')
                mask = pd.Series(True, index=filtered_df.index)  # Маску создаем с правильным индексом
                if min_val is not None:
                    mask &= filtered_df[col] >= min_val
                if max_val is not None:
                    mask &= filtered_df[col] <= max_val
                filtered_df = filtered_df[mask]
            else:
                # Категориальный фильтр: multiselect
                selected_values = config.get('selected', [])
                if selected_values:
                    filtered_df = filtered_df[filtered_df[col].isin(selected_values)]
    
    # Применение сортировки
    if sort_config:
        sort_col = sort_config.get('column')
        sort_order = sort_config.get('order', 'asc')
        if sort_col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[sort_col]):
            ascending = sort_order == 'asc'
            filtered_df = filtered_df.sort_values(by=sort_col, ascending=ascending)
    
    return filtered_df

def reset_filters():
    if st.session_state['original_df'] is not None:
        st.session_state['df'] = st.session_state['original_df'].copy()
        st.session_state.filters_applied = False
        st.session_state.history = [st.session_state['df'].copy()]  # Сброс истории к оригиналу
        st.session_state.current_step = 0
        st.success("Все фильтры сброшены, возвращена исходная таблица.")
        st.rerun()

# --- Главное меню ---
menu = st.sidebar.radio(
    "Меню",
    ["📊 Таблица", "📈 Статистика", "🔗 Корреляции", "📉 Гистограммы", "🧮 Кастомные метрики", "📂 Загрузка данных"]
)

# --- Загрузка данных ---
if menu == "📂 Загрузка данных":
    st.header("Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state['df'] = df.copy()
            st.session_state['original_df'] = df.copy()  # Исходная таблица
            st.session_state.history = [df.copy()]
            st.session_state.current_step = 0
            st.session_state.filters_applied = False
            st.success("Файл загружен и сохранён в сессии.")

# --- Работа с таблицей ---
elif st.session_state['df'] is not None:
    df = st.session_state['df']

    if menu == "📊 Таблица":
        st.header("Редактируемая таблица")
        st.write("Редактируйте таблицу или удалите столбец с помощью кнопки ниже.")

        # Флаг для включения фильтров и сортировки
        enable_filters_sort = st.checkbox("Включить фильтры и сортировку", value=st.session_state.filters_applied)

        # Определяем preview_df в зависимости от состояния
        if enable_filters_sort and st.session_state.filters_applied:
            preview_df = df  # Если фильтры уже применены, показываем текущий df
        elif enable_filters_sort:
            preview_df = apply_filters_and_sort(st.session_state['original_df'], st.session_state.get('filters', {}), st.session_state.get('sort_config', {}))
        else:
            preview_df = df  # Если фильтры выключены, показываем текущий df

        if enable_filters_sort:
            st.subheader("Фильтры и сортировка")
            
            # Инициализация словарей для фильтров и сортировки
            if 'filters' not in st.session_state:
                st.session_state.filters = {}
            if 'sort_config' not in st.session_state:
                st.session_state.sort_config = {}
            
            # Фильтры по столбцам (применяем к original_df)
            original_df_for_filters = st.session_state['original_df']
            for col in original_df_for_filters.columns:
                if col not in st.session_state.filters:
                    if pd.api.types.is_numeric_dtype(original_df_for_filters[col]):
                        st.session_state.filters[col] = {'min': None, 'max': None}
                    else:
                        st.session_state.filters[col] = {'selected': []}
                
                with st.expander(f"Фильтр для '{col}'"):
                    if pd.api.types.is_numeric_dtype(original_df_for_filters[col]):
                        min_val = st.number_input(f"Минимальное значение для {col}", value=None, key=f"min_{col}")
                        max_val = st.number_input(f"Максимальное значение для {col}", value=None, key=f"max_{col}")
                        st.session_state.filters[col] = {'min': min_val if min_val else None, 'max': max_val if max_val else None}
                    else:
                        selected = st.multiselect(f"Выберите значения для {col}", options=original_df_for_filters[col].unique(), default=st.session_state.filters[col]['selected'], key=f"multiselect_{col}")
                        st.session_state.filters[col] = {'selected': selected}
            
            # Сортировка (применяем к original_df)
            st.subheader("Сортировка")
            numeric_columns = original_df_for_filters.select_dtypes(include=['number']).columns.tolist()
            if numeric_columns:
                sort_col = st.selectbox("Выберите столбец для сортировки", options=numeric_columns, key="sort_col")
                sort_order = st.radio("Направление сортировки", options=["по возрастанию", "по убыванию"], key="sort_order")
                st.session_state.sort_config = {
                    'column': sort_col,
                    'order': 'asc' if sort_order == "по возрастанию" else 'desc'
                }
            else:
                st.warning("Нет числовых столбцов для сортировки.")
            
            # Кнопки применить и сбросить
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Применить фильтры и сортировку"):
                    filtered_sorted_df = apply_filters_and_sort(st.session_state['original_df'], st.session_state.filters, st.session_state.sort_config)
                    if not filtered_sorted_df.empty:
                        st.session_state['df'] = filtered_sorted_df.copy()
                        st.session_state.filters_applied = True
                        save_state(st.session_state['df'])
                        st.success("Фильтры и сортировка применены. Новая таблица стала основной.")
                        st.rerun()
                    else:
                        st.error("После применения фильтров таблица пуста. Измените фильтры.")
            
            with col2:
                if st.button("Сбросить все фильтры"):
                    reset_filters()

        # Редактируемая таблица (показываем основную или отфильтрованную)
        edited_df = st.data_editor(
            preview_df,
            num_rows="dynamic",
            use_container_width=True,
            key="editable_table"
        )
        if not edited_df.equals(preview_df):
            save_state(edited_df)
            st.success("Изменения в таблице сохранены.")

        # Удаление столбца (работает на основной таблице)
        st.write("Удалить столбец")
        column_to_delete = st.selectbox("Выберите столбец для удаления", df.columns, key="delete_column")
        if st.button("Удалить выбранный столбец"):
            if column_to_delete in df.columns:
                edited_df = df.drop(columns=[column_to_delete])
                save_state(edited_df)
                st.session_state['original_df'] = edited_df.copy()  # Обновляем original_df после удаления
                st.success(f"Столбец '{column_to_delete}' удалён.")
                st.rerun()  # Перерисовываем интерфейс для немедленного обновления
            else:
                st.error("Выбранный столбец не найден.")

        # Кнопка отмены действия
        if st.button("Отменить последнее действие"):
            undo_action()

    elif menu == "📈 Статистика":
        st.header("Описательная статистика")
        st.write(get_basic_stats(df))

    elif menu == "🔗 Корреляции":
        st.header("Корреляционная матрица")
        corr = get_correlations(df, method="pearson")
        if corr is not None:
            st.write(corr)

    elif menu == "📉 Гистограммы":
        st.header("Построение гистограммы")
        column = st.selectbox("Выберите колонку", df.columns)
        plot_histogram(df, column)

    elif menu == "🧮 Кастомные метрики":
        st.header("Кастомные метрики")
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

    # --- Кнопка скачать данные ---
    st.download_button(
        label="Скачать данные",
        data=df.to_csv(index=False).encode('utf-8'),
        file_name="data.csv",
        mime="text/csv"
    )

else:
    st.info("Перейдите во вкладку '📂 Загрузка данных' и загрузите CSV-файл.")