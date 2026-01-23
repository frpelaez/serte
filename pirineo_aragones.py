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
    df = pd.read_csv("./data/pirineo-aragones.csv")
    df.head()
    return


@app.cell
def _(TimeSeries):
    pirineo = (
        TimeSeries.from_csv("./data/pirineo-aragones.csv", "VALOR", "PERIODO")
        .with_name("Viajeros")
        .with_index_name("Tiempo")
    )
    pirineo.head()
    return (pirineo,)


@app.cell
def _(pirineo):
    pirineo.plot()
    return


if __name__ == "__main__":
    app.run()
