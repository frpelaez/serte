import pandas as pd

from timeseries import TimeSeries


def parse_date(date: str) -> str:
    trim = {1: "01", 2: "04", 3: "07", 4: "10"}
    q, y = date.split()
    num = int(q[1])
    return f"01-{trim[num]}-{y}"


def main():
    df = pd.read_spss("./data/gnp.sav")
    df["Tiempo"] = df["date_"].apply(parse_date)

    gnp = TimeSeries.from_dataframe(df, "indice", "Tiempo")
    gnp.plot()

    decomp = gnp.decompose(period=4)
    decomp.plot()

    nsdiffs = gnp.nsdiffs(period=4)
    print(nsdiffs)
    gnp_sdiff = gnp.diff(order=0, seasonal_lag=4, seasonal_order=nsdiffs)
    gnp_sdiff.plot()

    gnp.plot_correlograms()


if __name__ == "__main__":
    main()
