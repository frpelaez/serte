import marimo

__generated_with = "0.16.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from timeseries import TimeSeries
    return (TimeSeries,)


@app.cell
def _(TimeSeries):
    casas = (
        TimeSeries.from_spss("./data/casas.sav", "casas", "DATE_")
        .with_name("Casas")
        .with_index_name("Tiempo")
    )
    casas.plot()
    return (casas,)


@app.cell
def _(casas):
    decomp = casas.decompose(period=12)
    decomp.plot()
    return


@app.cell
def _(casas):
    casas.plot_correlograms()
    return


@app.cell
def _(casas):
    casas_bxcx, lmb = casas.box_cox()
    print("lambda:", lmb)
    casas_bxcx.plot()
    return (casas_bxcx,)


@app.cell
def _(casas_bxcx):
    casas_bxcx.nsdiffs(period=12)
    return


@app.cell
def _(casas_bxcx):
    casas_bxcx_sdiff = casas_bxcx.diff(order=0, seasonal_lag=12, seasonal_order=1)
    casas_bxcx_sdiff.plot()
    return (casas_bxcx_sdiff,)


@app.cell
def _(casas_bxcx_sdiff):
    casas_bxcx_sdiff.plot_correlograms()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""Tenemos que la gráfica de FAS decae lentamente, mientras que en la del FAP tenemos un corte abrupto después del lag 1. Con la diferenciación que habíamos hecho, tenemos d = 1 y p = 1.""")
    return


if __name__ == "__main__":
    app.run()
