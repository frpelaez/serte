import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    import marimo as mo
    from timeseries import TimeSeries
    import pandas as pd
    return TimeSeries, mo, pd


@app.cell
def _(pd):
    df = pd.read_spss("./data/gnp.sav")


    def parse_date(date: str) -> str:
        trim = {1: "01", 2: "04", 3: "07", 4: "10"}
        q, y = date.split()
        num = int(q[1])
        return f"01-{trim[num]}-{y}"


    df["Tiempo"] = df["date_"].apply(parse_date)
    print(df.head())
    return (df,)


@app.cell
def _(TimeSeries, df):
    gnp = TimeSeries.from_dataframe(df, "indice", "Tiempo")
    gnp.plot()
    return (gnp,)


@app.cell
def _(gnp):
    decomp = gnp.decompose(period=4)
    decomp.plot()
    return


@app.cell
def _(gnp):
    gnp.plot_correlograms()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    Vemos que la gráfica de FAS decae muy lentamente, esto se corresponde con la tendencia que observamos en la gráfica.

    En la gráfica de FAP vemos una correlación casi total en el lag 1 y después barras no significativas, parece un camino aleatorio.
    """
    )
    return


@app.cell
def _(gnp):
    gnp_est = gnp.log().diff(order=0, seasonal_lag=4, seasonal_order=1)
    gnp_est.plot()
    return (gnp_est,)


@app.cell
def _(mo):
    mo.md(
        r"""Al aplicar una transformación logarítmica y luego diferenciar estacionalmente una vez hemos eliminazo por completo la tencendia, por lo que es casi seguro que nuestro parámetro de diferenciación sea d = 1."""
    )
    return


@app.cell
def _(gnp_est):
    gnp_est.plot_correlograms()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    En el nuevo gráfico FAS vemos un decaimiento más rápido, aunque no abrupto, con barras significativas en los lags 1, 2 y más o menos 3. Este decaimiento nos sugiere q = 0, es decir, sin componente MA.

    En la gráfica FAP vemos barras significativas en los primeros 3 lags. Sin embargo, la barra correspondiente al lag 3 es significativa por bastante poco, sobre todo en comparación con los lags 1 y 2. Por ello prefiero como modelo candidato un AR(2).
    """
    )
    return


if __name__ == "__main__":
    app.run()
