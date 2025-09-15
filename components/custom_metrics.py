import pandas as pd
import io

def compute_custom_metric(df, code_str):
    """Выполняет пользовательский код с доступом к df и стандартным библиотекам."""
    local_vars = {'df': df} if df is not None else {}
    result = None
    try:
        # Ограничиваем доступ к безопасным встроенным функциям
        safe_builtins = {'__builtins__': {'len': len, 'range': range, 'print': print, 'int': int, 'float': float, 'str': str, 'dict': dict}}
        exec_locals = local_vars.copy()
        
        # Выполняем код
        exec(code_str, safe_builtins, exec_locals)
        
        # Проверяем результат
        if 'result' in exec_locals:
            result = exec_locals['result']
        elif 'result' in local_vars:
            result = local_vars['result']
        
        if result is None:
            return "Код выполнен, но результат не задан (используйте 'result = ...')."
        elif isinstance(result, (pd.DataFrame, pd.Series)):
            return result
        return str(result)
    except AttributeError as e:
        return f"Ошибка: {str(e)} - Проверьте, не используете ли вы метод .dict() для словаря напрямую."
    except Exception as e:
        return f"Ошибка: {str(e)}"

def export_result(result):
    """Экспортирует результат в CSV или текст в зависимости от типа."""
    if isinstance(result, (pd.DataFrame, pd.Series)):
        output = result.to_csv(index=False)
        return output.encode('utf-8'), "result.csv", "text/csv"
    elif isinstance(result, str):
        return result.encode('utf-8'), "result.txt", "text/plain"
    return str(result).encode('utf-8'), "result.txt", "text/plain"