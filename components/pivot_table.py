import pandas as pd

def build_pivot_table(df, index, columns, values, aggfunc):
    """Строит сводную таблицу на основе указанных параметров."""
    try:
        pivot_df = pd.pivot_table(df, values=values, index=index, columns=columns, aggfunc=aggfunc, fill_value=0)
        return pivot_df
    except Exception as e:
        return f"Ошибка при построении сводной таблицы: {str(e)}"