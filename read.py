from typing import Optional

import numpy as np
import pandas as pd
import requests

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


def read_spss(
    filepath: str,
    value_col_name: str,
    index_col_name: str,
    convert_categorical: bool = True,
) -> TimeSeries:
    return TimeSeries.from_spss(
        filepath, value_col_name, index_col_name, convert_categorical
    )


def http_unindexed(
    url: str | bytes, val_col_name: str, index: Optional[pd.DatetimeIndex] = None
) -> TimeSeries:
    response = requests.get(url)
    if response.status_code == 200:
        data = np.array(
            list(
                map(
                    float,
                    filter(lambda s: len(s) > 0, response.content.splitlines()),
                )
            )
        )
    else:
        raise requests.HTTPError(
            f"Request failed with status code {response.status_code}"
        )
    data = pd.Series(data)
    if index:
        data.index = pd.to_datetime(index)
    return TimeSeries(data, val_col_name)


def http_indexed(url: str | bytes, val_col_name: str, sep: str = ",") -> TimeSeries:
    response = requests.get(url)
    if response.status_code == 200:
        print(response.text)
        data = np.array(
            list(
                map(
                    lambda pair: [pd.to_datetime(pair[0]), float(pair[1])],
                    map(
                        lambda s: s.decode().split(sep),
                        filter(lambda s: len(s) > 0, response.content.splitlines()),
                    ),
                )
            )
        )
    else:
        raise requests.HTTPError(
            f"Request failed with status code {response.status_code}"
        )
    data = pd.Series(data)
    return TimeSeries(data, val_col_name)


def _test():
    # srs_csv = read_csv("data/data.csv", "sells", "date")
    # print(srs_csv, srs_csv.data.head())
    # srs_xlsx = read_excel("data/data.xlsx", "sells", "date")
    # print(srs_xlsx, srs_xlsx.data.head())
    # srs_json = read_json("data/data.json", "bitcoin", "date")
    # print(srs_json, srs_json.data.head())
    # url = "http://verso.mat.uam.es/~joser.berrendero/datos/gas6677.dat"
    # srs_http = http_unindexed(url, "srs")
    # print(srs_http, srs_http.head())

    df = pd.read_csv("data/pirineos.csv", sep=";")
    df = df[(df["Valor3"] == "Pirineus") & (df["VALOR"] != ".")]
    df = df[["PERIODO", "VALOR"]]
    df["PERIODO"] = df["PERIODO"].map(lambda x: f"{x[-2:]}-{x[:4]}")
    df["VALOR"] = df["VALOR"].map(lambda x: int(x.replace(".", "")))

    pirineos = TimeSeries.from_dataframe(df, "VALOR", "PERIODO").rename(
        "Pernoctaciones en Pirineos"
    )
    print(pirineos.head())
    pirineos.plot()


if __name__ == "__main__":
    _test()
