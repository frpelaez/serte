from core.timeseries import TimeSeries


def main():
    path = "./data/daily-minimum-temperatures-in-me.csv"
    st = TimeSeries.from_csv(path, "Daily minimum temperatures", "Date")
    print(st.data.head())
    st = st.to_numeric()
    print(st.head())
    st.plot()

    diff_est = st.diff(seasonal_lag=30, seasonal_order=1)
    print(diff_est.head())
    diff_est.plot()


if __name__ == "__main__":
    main()
