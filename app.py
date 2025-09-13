import streamlit as st
import pandas as pd
from utils.data_loader import load_data
from utils.stats import get_extended_stats, detect_outliers, get_correlations
from visualizations.plots import plot_histogram, plot_boxplot, plot_scatter, plot_line, plot_bar
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
if "prev_stats" not in st.session_state:
    st.session_state['prev_stats'] = None

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

def handle_missing_values(df, missing_info):
    # Проверяем наличие пропусков
    total_missing = missing_info["total_missing"]
    if total_missing == 0:
        return df

    st.warning(f"Обнаружено {total_missing} пропусков в датасете. Настройте обработку ниже.")
    
    # Разделяем столбцы на числовые и категориальные
    numeric_cols = missing_info["numeric_cols"]
    categorical_cols = missing_info["categorical_cols"]

    # Обработка для числовых столбцов
    if numeric_cols:
        with st.expander("Обработка пропусков в числовых столбцах", expanded=True):
            st.write(f"Числовые столбцы: {', '.join(numeric_cols)}")
            numeric_action = st.selectbox("Действие для числовых пропусков", 
                                        ["Удалить строки с пропусками", "Оставить пропуски", 
                                         "Заменить на среднее", "Заменить на медиану", "Заменить на моду"])
            
            if numeric_action == "Удалить строки с пропусками":
                df = df.dropna(subset=numeric_cols)
            elif numeric_action == "Заменить на среднее":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            elif numeric_action == "Заменить на медиану":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            elif numeric_action == "Заменить на моду":
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mode().iloc[0])

    # Обработка для категориальных столбцов
    if categorical_cols:
        with st.expander("Обработка пропусков в категориальных столбцах", expanded=True):
            st.write(f"Категориальные столбцы: {', '.join(categorical_cols)}")
            categorical_action = st.selectbox("Действие для категориальных пропусков", 
                                            ["Оставить пропуски", "Удалить строки с пропусками"])
            
            if categorical_action == "Удалить строки с пропусками":
                df = df.dropna(subset=categorical_cols)

    st.success("Обработка пропусков завершена.")
    return df

# --- Главное меню ---
menu = st.sidebar.radio(
    "Меню",
    ["📊 Таблица", "📈 Статистика", "📊 Визуализация", "🧮 Кастомные метрики", "📂 Загрузка данных"]
)

# --- Загрузка данных ---
if menu == "📂 Загрузка данных":
    st.header("Загрузка данных")
    uploaded_file = st.file_uploader("Загрузите CSV файл", type=["csv"])
    if uploaded_file is not None:
        df, missing_info = load_data(uploaded_file)
        if df is not None:
            # Проверяем и обрабатываем пропуски
            df = handle_missing_values(df, missing_info)
            st.session_state['df'] = df.copy()
            st.session_state['original_df'] = df.copy()  # Исходная таблица
            st.session_state.history = [df.copy()]
            st.session_state.current_step = 0
            st.session_state.filters_applied = False
            st.session_state['prev_stats'] = None  # Сброс предыдущей статистики
            st.success("Файл загружен и обработан (пропуски устранены).")

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
        
        # Выбор столбцов для анализа
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        selected_cols = st.multiselect("Выберите столбцы для анализа", numeric_cols, default=numeric_cols)
        
        # Фильтр для статистики
        col1, col2 = st.columns(2)
        with col1:
            filter_col = st.selectbox("Фильтровать по столбцу", ["Нет"] + df.columns.tolist())
        with col2:
            filter_value = st.number_input("Значение фильтра", value=None, key="filter_value") if filter_col != "Нет" and pd.api.types.is_numeric_dtype(df[filter_col]) else None
        
        # Применение фильтра
        filtered_df = df.copy()
        if filter_col != "Нет" and filter_value is not None:
            filtered_df = filtered_df[filtered_df[filter_col] >= filter_value]
        
        # Расширенная статистика
        if selected_cols:
            stats_df = get_extended_stats(filtered_df, selected_cols)
            # Применяем форматирование только к числовым столбцам, исключая 'Столбец'
            numeric_cols_in_stats = [col for col in stats_df.columns if col != 'Столбец']
            styled_df = stats_df.style.format({col: "{:.2f}" for col in numeric_cols_in_stats}).background_gradient(cmap='Blues')
            st.dataframe(styled_df)
        
            # Визуализация (Boxplot)
            fig = plot_boxplot(filtered_df, selected_cols)
            st.plotly_chart(fig, use_container_width=True)
        
            # Обнаружение аномалий (только количество)
            outliers = detect_outliers(filtered_df, selected_cols)
            if any(outliers.values()):
                st.subheader("Количество выбросов")
                for col, indices in outliers.items():
                    if indices:
                        count = len(indices)
                        st.write(f"Столбец {col}: {count} выбросов")
        
            # Heatmap корреляций
            corr = get_correlations(filtered_df)
            if corr is not None and len(corr.columns) > 1:
                st.subheader("Тепловая карта корреляций")
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale='RdBu_r')
                fig.update_layout(title="Корреляционная матрица")
                st.plotly_chart(fig, use_container_width=True)
        
        # Сравнение с предыдущим состоянием
        if st.session_state['prev_stats'] is not None and selected_cols:
            st.subheader("Сравнение с предыдущим состоянием")
            prev_stats_df = st.session_state['prev_stats'][st.session_state['prev_stats']['Столбец'].isin(selected_cols)]
            diff_df = pd.merge(stats_df, prev_stats_df, on="Столбец", suffixes=('_new', '_old'))
            diff_df['Разница (Среднее)'] = diff_df['Среднее_new'] - diff_df['Среднее_old']
            styled_diff = diff_df.style.format({col: "{:.2f}" for col in diff_df.columns if col != 'Столбец'}).background_gradient(cmap='RdYlGn', subset=['Разница (Среднее)'])
            st.dataframe(styled_diff)
        
        # Кастомные метрики
        custom_formula = st.text_area("Введите кастомную формулу (например, df['age'] * 2)", 
                                    placeholder="Пример: df['age'] * 2 или df['salary'] / 1000")
        if st.button("Вычислить кастомную метрику"):
            if custom_formula and selected_cols:
                try:
                    custom_df = filtered_df.copy()
                    custom_df['Custom'] = eval(custom_formula)
                    custom_stats = get_extended_stats(custom_df, ['Custom'])
                    styled_custom = custom_stats.style.format({col: "{:.2f}" for col in custom_stats.columns if col != 'Столбец'})
                    st.write("Статистика кастомной метрики:")
                    st.dataframe(styled_custom)
                except Exception as e:
                    st.error(f"Ошибка в формуле: {e}")

        # Сохранение текущей статистики
        if st.button("Сохранить текущее состояние статистики"):
            st.session_state['prev_stats'] = get_extended_stats(df, selected_cols)
            st.success("Статистика сохранена.")

    elif menu == "📊 Визуализация":
        st.header("Визуализация данных")
        
        # Выбор типа визуализации
        chart_types = ["Гистограмма", "Ящик с усами", "Точечная диаграмма", "Линейный график", "Столбчатая диаграмма"]
        chart_type = st.selectbox("Выберите тип визуализации", chart_types)
        
        # Выбор столбцов
        all_cols = df.columns.tolist()
        if chart_type in ["Точечная диаграмма", "Линейный график"]:
            x_col = st.selectbox("Выберите столбец для оси X", all_cols)
            y_col = st.selectbox("Выберите столбец для оси Y", [col for col in all_cols if col != x_col])
            selected_cols = [x_col, y_col]
        else:
            selected_cols = st.multiselect("Выберите столбцы для визуализации", all_cols, default=all_cols[0] if all_cols else None)
        
        # Фильтр для визуализации
        col1, col2 = st.columns(2)
        with col1:
            filter_col = st.selectbox("Фильтровать по столбцу", ["Нет"] + df.columns.tolist())
        with col2:
            filter_value = st.number_input("Значение фильтра", value=None, key="viz_filter_value") if filter_col != "Нет" and pd.api.types.is_numeric_dtype(df[filter_col]) else None
        
        # Применение фильтра
        viz_df = df.copy()
        if filter_col != "Нет" and filter_value is not None:
            viz_df = viz_df[viz_df[filter_col] >= filter_value]
        
        # Настройки
        bins = st.slider("Количество бинов (для гистограммы)", 10, 50, 30) if chart_type == "Гистограмма" else None
        color_col = st.selectbox("Цвет по столбцу (опционально)", ["Нет"] + all_cols) if chart_type in ["Гистограмма", "Точечная диаграмма", "Столбчатая диаграмма"] else None
        
        # Генерация графика
        try:
            if selected_cols and any(viz_df[col].notna().any() for col in selected_cols):
                if chart_type == "Гистограмма":
                    fig = plot_histogram(viz_df, selected_cols[0], nbins=bins, color_col=color_col if color_col != "Нет" else None)
                elif chart_type == "Ящик с усами":
                    fig = plot_boxplot(viz_df, selected_cols)
                elif chart_type == "Точечная диаграмма":
                        fig = plot_scatter(viz_df, x_col, y_col, color_col=color_col if color_col != "Нет" else None)
                elif chart_type == "Линейный график":
                    fig = plot_line(viz_df, x_col, y_col)
                elif chart_type == "Столбчатая диаграмма":
                    fig = plot_bar(viz_df, selected_cols[0], color_col=color_col if color_col != "Нет" else None)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Выберите хотя бы один столбец с данными.")
        except:
            st.write("Недопустимые данные")


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