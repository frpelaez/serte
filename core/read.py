from timeseries import TimeSeries


def read_csv(
    filepath: str, value_col_name: str, index_col_name: str, **kwargs
) -> TimeSeries:
    return TimeSeries.from_csv(
        filepath, value_col_name=value_col_name, index_col_name=index_col_name, **kwargs
    )


def read_excel(
    filepath: str,
    value_col_name: str,
    index_col_name: str,
    sheet_name: int | str = 0,
    **kwargs,
) -> TimeSeries:
    return TimeSeries.from_excel(
        filepath,
        value_col_name=value_col_name,
        index_col_name=index_col_name,
        sheet_name=sheet_name,
        **kwargs,
    )


def read_json(
    filepath: str, value_col_name: str, index_col_name: str, **kwargs
) -> TimeSeries:
    return TimeSeries.from_json(
        filepath, value_col_name=value_col_name, index_col_name=index_col_name, **kwargs
    )


def _test():
    srs_csv = read_csv("data/data.csv", "sells", "date")
    print(srs_csv, srs_csv.data.head())
    srs_xlsx = read_excel("data/data.xlsx", "sells", "date")
    print(srs_xlsx, srs_xlsx.data.head())
    srs_json = read_json("data/data.json", "bitcoin", "date")
    print(srs_json, srs_json.data.head())


if __name__ == "__main__":
    _test()
