from timeseries import TimeSeries

import matplotlib.pyplot as plt


def main():
    casas = (
        TimeSeries.from_spss("./data/casas.sav", "casas", "DATE_")
        .with_name("Casas")
        .with_index_name("Tiempo")
    )
    casas.plot()

    decomp = casas.decompose()
    decomp.plot()

    casas_boxcox, lmb = casas.box_cox()
    print(lmb)
    casas_boxcox.plot()

    nsdiffs = casas_boxcox.nsdiffs(period=12, test="seas")
    print(nsdiffs)
    casas_boxcox_diff = casas_boxcox.diff(nsdiffs, seasonal_lag=12)
    casas_boxcox_diff.plot()

    nsdiffs2 = casas_boxcox_diff.nsdiffs(period=12, test="seas")
    print(nsdiffs2)
    casas_boxcox_diff2 = casas_boxcox_diff.diff(nsdiffs2, seasonal_lag=12)
    casas_boxcox_diff2.plot()

    casas_sma = casas.smooth_moving_avg(window=6)
    casas_exp = casas.smooth_exponential(span=6)
    casas_hw = casas.smooth_holt_winters(trend="add", seasonal=None)

    plt.figure(figsize=(12, 6))

    plt.plot(casas.data, label="Original", color="lightgray", linestyle="-")

    plt.plot(casas_sma.data, label="SMA (Simple)", color="blue", linewidth=2)
    plt.plot(casas_exp.data, label="EMA (Exponential)", color="orange", linestyle="--")
    plt.plot(casas_hw.data, label="Holt-Winters", color="red", linestyle="-.")

    plt.title("Comparación de Técnicas de Suavizado")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
