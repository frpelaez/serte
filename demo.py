from core.timeseries import TimeSeries


def main():
    path = "./data/daily-minimum-temperatures-in-me.csv"
    st = TimeSeries.from_csv(path, "Daily minimum temperatures", "Date")
    print(st.data.head())

    st = st.to_numeric()
    print(st.count_na(), st.count_null())
    st = st.replace_na(st.data.mean())
    print(st.count_na())
    st.plot()

    st = st.map(lambda x: x + 273).rename("min temp kelvin")
    box_cox_st, lmd = st.box_cox()
    print(f"{lmd=:.3f}")
    print(box_cox_st.head())
    box_cox_st.plot()

    diff_est = st.diff(seasonal_lag=30, seasonal_order=1)
    print(diff_est.head())
    diff_est.plot()


if __name__ == "__main__":
    main()
