import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium", auto_download=["ipynb"])


@app.cell
def _():
    from timeseries import TimeSeries
    import pandas as pd
    return TimeSeries, pd


@app.cell
def _(pd):
    df = pd.read_spss("./data/total_acciones.sav")
    df.head()
    return


@app.cell
def _(TimeSeries):
    acciones = (
        TimeSeries.from_spss("./data/total_acciones.sav", "Total_acciones", "DATE_")
        .with_name("Acciones")
        .with_index_name("Tiempo")
    )
    acciones.plot()
    return (acciones,)


@app.cell
def _(acciones):
    acciones_log = acciones.log()
    acciones_log.plot()
    return (acciones_log,)


@app.cell
def _(acciones_log):
    acciones_log.nsdiffs(period=12)
    return


@app.cell
def _(acciones_log):
    acciones_log_diff = acciones_log.diff()
    acciones_log_diff.plot()
    return (acciones_log_diff,)


@app.cell
def _(acciones_log_diff):
    acciones_log_diff.plot_correlograms(lags=48)
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    p = 2, P = 1

    d = 1, D = 0

    q = 1, Q = 2
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
