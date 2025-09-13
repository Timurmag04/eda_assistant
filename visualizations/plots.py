import plotly.express as px

def plot_histogram(df, x_col, nbins=None, color_col=None, title=None):
    """Создает гистограмму для указанного столбца."""
    fig = px.histogram(df, x=x_col, nbins=nbins, color=color_col if color_col else None, title=title or f"Гистограмма: {x_col}")
    fig.update_layout(showlegend=True)
    return fig

def plot_boxplot(df, y_cols, title=None):
    """Создает ящик с усами для указанных столбцов."""
    fig = px.box(df, y=y_cols, title=title or "Ящик с усами")
    fig.update_layout(showlegend=True)
    return fig

def plot_scatter(df, x_col, y_col, color_col=None, title=None):
    """Создает точечную диаграмму с опциональной регрессионной линией."""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col if color_col else None, trendline="ols", title=title or f"Точечная диаграмма: {x_col} vs {y_col}")
    fig.update_layout(showlegend=True)
    return fig

def plot_line(df, x_col, y_col, title=None):
    """Создает линейный график."""
    fig = px.line(df, x=x_col, y=y_col, title=title or f"Линейный график: {x_col} vs {y_col}")
    fig.update_layout(showlegend=True)
    return fig

def plot_bar(df, x_col, color_col=None, title=None):
    """Создает столбчатую диаграмму."""
    fig = px.bar(df, x=x_col, color=color_col if color_col else None, title=title or f"Столбчатая диаграмма: {x_col}")
    fig.update_layout(showlegend=True)
    return fig