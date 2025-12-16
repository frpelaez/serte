from typing import Callable, Generator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats


class TimeSeries:
    def __init__(self, data: pd.Series, name: str = "series") -> TimeSeries:
        if not isinstance(data, pd.Series):
            raise TypeError("Provided data must be a pandas Series")
        self.data = data.sort_index()
        self.data.name = name
        self.name = name

    def __repr__(self):
        return f"<TimeSeries: {self.name} | size={len(self.data)} | from {self.data.index.min()} up to {self.data.index.max()}>"

    @staticmethod
    def from_dataframe(
        df: pd.DataFrame, value_col_name: str, index_col_name: Optional[str] = None
    ) -> TimeSeries:
        if index_col_name:
            df = df.set_index(index_col_name)
            df.index = pd.to_datetime(df.index)
        return TimeSeries(df[value_col_name], value_col_name)

    @staticmethod
    def from_csv(
        filepath: str, value_col_name: str, index_col_name: str, **kwargs
    ) -> TimeSeries:
        df = pd.read_csv(
            filepath, parse_dates=[index_col_name], index_col=index_col_name, **kwargs
        )
        return TimeSeries(df[value_col_name], value_col_name)

    @staticmethod
    def from_excel(
        filepath: str, value_col_name: str, index_col_name: str, sheet_name=0, **kwargs
    ) -> TimeSeries:
        df = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            parse_dates=[index_col_name],
            index_col=index_col_name,
            **kwargs,
        )
        return TimeSeries(df[value_col_name], value_col_name)

    @staticmethod
    def from_json(
        filepath: str, value_col_name: str, index_col_name: str, **kwargs
    ) -> TimeSeries:
        df = pd.read_json(filepath, convert_dates=False, **kwargs)
        if index_col_name in df.columns:
            df = df.set_index(index_col_name)
        df.index = pd.to_datetime(df.index)
        return TimeSeries(df[value_col_name], value_col_name)

    def to_pandas(self) -> pd.Series:
        return self.data

    def to_list(self) -> list[tuple[...]]:
        return list(zip(self.times, self.values))

    def get_times(self) -> pd.DatetimeIndex:
        return self.data.index

    @property
    def times(self) -> pd.DatetimeIndex:
        return self.get_times()

    def get_values(self) -> np.ndarray:
        return self.data.values

    @property
    def values(self) -> np.ndarray:
        return self.get_values()

    def head(self) -> pd.Series:
        return self.data.head()

    def min(self) -> float:
        if not (isinstance(self.values[0], int) or isinstance(self.values[0], float)):
            raise TypeError(
                "Values of series are non-numeric, unnable to evaluate `min`/`max`. Consider casting to numeric values: `.to_numeric()`"
            )
        return self.data.min()

    def max(self) -> float:
        if not (isinstance(self.values[0], int) or isinstance(self.values[0], float)):
            raise TypeError(
                "Values of series are non-numeric, unnable to evaluate `min`/`max`. Consider casting to numeric values: `.to_numeric()`"
            )
        return self.data.max()

    def size(self) -> int:
        return len(self.data)

    def __len__(self) -> int:
        return self.size()

    def log(self) -> TimeSeries:
        if (self.data <= 0).any():
            raise ValueError(
                "Series contains negative values, unnable to apply 'log' transformation"
            )
        data = np.log(self.data)
        return TimeSeries(data, f"log_{self.name}")

    def sqrt(self) -> TimeSeries:
        if (self.data <= 0).any():
            raise ValueError(
                "Series contains negative values, unnable to apply 'sqrt' transformation"
            )
        data = np.sqrt(self.data)
        return TimeSeries(data, f"sqrt_{self.name}")

    def inv(self) -> TimeSeries:
        if (self.data == 0).any():
            raise ValueError(
                "Series contains null values, unnable to apply 'inv' transformation"
            )
        data = 1 / self.data
        return TimeSeries(data, f"inv_{self.name}")

    def pow(self, power: float) -> TimeSeries:
        if power < 0 and (self.data <= 0).any():
            raise ValueError(
                "Provided power is negative and series contains negative values, unnable to apply transformation"
            )
        data = np.power(self.data, power)
        return TimeSeries(data, f"pow{power}_{self.name}")

    def box_cox(self, lmd: Optional[float] = None) -> tuple[TimeSeries, float]:
        if (self.data <= 0).any():
            raise ValueError(
                "Series contains negative values, unnable to apply 'box-cox' transformation"
            )
        if lmd is None:
            data, best_lambda = stats.boxcox(self.data)
            name = f"box-cox_{best_lambda:.2f}_{self.name}"
        else:
            data = stats.boxcox(self.data, lmd)
            best_lambda = lmd
            name = f"box-cox_{lmd:.2f}_{self.name}"
        series = pd.Series(data, index=self.data.index)
        return TimeSeries(series, name), best_lambda

    def rename(self, name: str) -> TimeSeries:
        return TimeSeries(self.data.copy(), name)

    def to_numeric(self, errors: str = "coerce") -> TimeSeries:
        data = pd.to_numeric(self.data, errors=errors)
        return TimeSeries(data, self.name)

    def count_na(self) -> int:
        return self.data.isna().sum()

    def count_null(self) -> int:
        return self.data.isnull().sum()

    def mean(self) -> float:
        return self.data.mean()

    def median(self) -> float:
        return self.data.median()

    def variance(self) -> float:
        return self.data.var()

    def stddev(self) -> float:
        return self.data.std()

    def replace_na(self, value: float) -> TimeSeries:
        data = self.data.fillna(value)
        return TimeSeries(data, self.name)

    def map(self, f: Callable[[pd.Series], pd.Series]) -> TimeSeries:
        data = f(self.data)
        return TimeSeries(data, f"{f.__name__}_{self.name}")

    def iter_pairs(self) -> Generator[tuple[...], None, None]:
        for d, v in zip(self.times, self.values):
            yield d, v

    def diff(
        self, order: int = 1, seasonal_lag: int = 0, seasonal_order: int = 0
    ) -> TimeSeries:
        temp = self.data.copy()
        if seasonal_lag > 0 and seasonal_order > 0:
            for _ in range(seasonal_order):
                temp = temp.diff(periods=seasonal_lag)
        if order > 0:
            for _ in range(order):
                temp = temp.diff(periods=1)
        temp = temp.dropna()
        suffix = f"_Diff{order}"
        if seasonal_order > 0:
            suffix += f"_Sord{seasonal_order}_Slag{seasonal_lag}"
        return TimeSeries(temp, f"{self.name}{suffix}")

    def plot(self):
        self.data.plot(title=self.name)
        plt.show()


def _test():
    # Leer de csv (también se puede de pd.DataFrame, excel o json)
    path = "data/data.csv"
    series = TimeSeries.from_csv(path, value_col_name="sells", index_col_name="date")

    # Info general y datos
    print("\n- Info general y datos internos -")
    print(series)
    print(series.data)

    # Transformaciones
    print("\n- Transformaciones -")
    print(series.log().data)
    print(series.sqrt().data)
    print(series.inv().data)
    print(series.pow(2).data)
    print(series.box_cox()[0].data)
    print(series.map(lambda x: 3 * x - x**2).data)

    def mi_transformacion(x):
        return 1 / np.sqrt(x)

    print(series.map(mi_transformacion).data)

    # Iterar sobre parejas (fecha, valor)
    print("\n- Iterar sobre fechas, valores o parejas fecha-valor")
    for d, v in series.iter_pairs():
        print(f"Fecha: {d} | Valor: {v}")

    # Acceder a fechas/valores de forma cómoda
    print("\n- Acceder a fechas y valores -")
    print(series.times)
    print(series.values)

    # Diferenciar la serie
    print("\n- Diferenciar la serie -")
    print(series.diff(order=1).data)
    print(series.diff(order=2).data)

    # Plot rápido
    series.plot()


if __name__ == "__main__":
    _test()
