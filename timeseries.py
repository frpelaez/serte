from typing import Callable, Generator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf

from decomposition import DecompositionResult


class TimeSeries:
    def __init__(self, data: pd.Series, name: str = "series") -> 'TimeSeries':
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
    ) -> 'TimeSeries':
        if index_col_name:
            df = df.set_index(index_col_name)
            df.index = pd.to_datetime(df.index)
        return TimeSeries(df[value_col_name], value_col_name)

    @staticmethod
    def from_csv(
        filepath: str, value_col_name: str, index_col_name: str, **kwargs
    ) -> 'TimeSeries':
        df = pd.read_csv(
            filepath, parse_dates=[index_col_name], index_col=index_col_name, **kwargs
        )
        return TimeSeries(df[value_col_name], value_col_name)

    @staticmethod
    def from_excel(
        filepath: str, value_col_name: str, index_col_name: str, sheet_name=0, **kwargs
    ) -> 'TimeSeries':
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
    ) -> 'TimeSeries':
        df = pd.read_json(filepath, convert_dates=False, **kwargs)
        if index_col_name in df.columns:
            df = df.set_index(index_col_name)
        else:
            raise ValueError(
                f"Column '{index_col_name}' does not exit in the provided data"
            )
        df.index = pd.to_datetime(df.index)
        return TimeSeries(df[value_col_name], value_col_name)

    @staticmethod
    def from_spss(
        filepath: str,
        value_col_name: str,
        index_col_name: str,
        convert_categoricals: bool = True,
    ) -> 'TimeSeries':
        df = pd.read_spss(filepath, convert_categoricals=convert_categoricals)
        if index_col_name in df.columns:
            df = df.set_index(index_col_name)
        else:
            raise ValueError(
                f"Column '{index_col_name}' does not exit in the provided data"
            )
        df.index = pd.to_datetime(df.index)
        return TimeSeries(df[value_col_name], name=value_col_name)

    def to_csv(self, output: str, **kwargs):
        self.data.to_csv(output, **kwargs)
        print(f"TimeSeries successfully written to {output}")

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

    def log(self) -> 'TimeSeries':
        if (self.data <= 0).any():
            raise ValueError(
                "Series contains negative values, unnable to apply 'log' transformation"
            )
        data = np.log(self.data)
        return TimeSeries(data, f"log_{self.name}")

    def sqrt(self) -> 'TimeSeries':
        if (self.data <= 0).any():
            raise ValueError(
                "Series contains negative values, unnable to apply 'sqrt' transformation"
            )
        data = np.sqrt(self.data)
        return TimeSeries(data, f"sqrt_{self.name}")

    def inv(self) -> 'TimeSeries':
        if (self.data == 0).any():
            raise ValueError(
                "Series contains null values, unnable to apply 'inv' transformation"
            )
        data = 1 / self.data
        return TimeSeries(data, f"inv_{self.name}")

    def pow(self, power: float) -> 'TimeSeries':
        if power < 0 and (self.data <= 0).any():
            raise ValueError(
                "Provided power is negative and series contains negative values, unnable to apply transformation"
            )
        data = np.power(self.data, power)
        return TimeSeries(data, f"pow{power}_{self.name}")

    def box_cox(self, lmd: Optional[float] = None) -> tuple['TimeSeries', float]:
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
        if abs(best_lambda - 1.0) <= 0.05:
            print(
                "WARN: lambda value close to 1. Consider discarding the transformation"
            )
        return TimeSeries(series, name), best_lambda

    def with_name(self, name: str) -> 'TimeSeries':
        data = self.data.copy()
        data.columns = [name]
        return TimeSeries(data, name)

    def with_index_name(self, index_name) -> 'TimeSeries':
        data = self.data.copy()
        data.index.name = index_name
        return TimeSeries(data, self.name)

    def with_index(self, index: pd.DatetimeIndex) -> 'TimeSeries':
        index = pd.to_datetime(index)
        self.data.index = index
        return TimeSeries(self.data, self.name)

    def to_numeric(self, errors: str = "coerce") -> 'TimeSeries':
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

    def replace_na(self, value: float) -> 'TimeSeries':
        data = self.data.fillna(value)
        return TimeSeries(data, self.name)

    def apply(self, f: Callable[[pd.Series], pd.Series]) -> 'TimeSeries':
        data = f(self.data)
        return TimeSeries(data, f"{f.__name__}_{self.name}")

    def iter_pairs(self) -> Generator[tuple[...], None, None]:
        for d, v in zip(self.times, self.values):
            yield d, v

    def diff(
        self, order: int = 1, seasonal_lag: int = 0, seasonal_order: int = 0
    ) -> 'TimeSeries':
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

    def moving_avg(self, kernel_range: int) -> 'TimeSeries':
        temp = np.zeros(shape=(2 * kernel_range + len(self.data),))
        temp[kernel_range:-kernel_range] = self.data.copy()
        temp[:kernel_range] = np.array([self.data.iloc[0] for _ in range(kernel_range)])
        temp[-kernel_range:] = np.array(
            [self.data.iloc[-1] for _ in range(kernel_range)]
        )
        kernel = np.ones(shape=(2 * kernel_range + 1,))
        kernel /= kernel.sum()
        for i in range(kernel_range, len(temp) - kernel_range):
            temp[i] = temp[i - kernel_range : i + kernel_range + 1].dot(kernel)
        temp = temp[kernel_range:-kernel_range]
        data = pd.Series(temp, self.data.index)
        return TimeSeries(data, f"mov_avg_{kernel_range}_{self.name}")

    def smooth_moving_avg(self, window: int, center: bool = False) -> 'TimeSeries':
        data = self.data.rolling(window=window, center=center).mean()
        data = data.dropna()
        return TimeSeries(data, f"SMA_{window}_{self.name}")

    def smooth_exponential(
        self, span: Optional[int] = None, alpha: Optional[float] = None
    ) -> 'TimeSeries':
        if span:
            data = self.data.ewm(span=span).mean()
            suffix = f"span{span}"
        else:
            data = self.data.ewm(alpha=alpha).mean()
            suffix = f"alpha{alpha}"
        return TimeSeries(data, f"ExpMA_{suffix}_{self.name}")

    def smooth_holt_winters(
        self,
        trend: Optional[str] = None,
        seasonal: Optional[str] = None,
        seasonal_periods: Optional[int] = None,
    ) -> 'TimeSeries':
        data = self.data
        if data.index.freq is None:
            try:
                data.index.freq = pd.infer_freq(data.index)
            except:
                pass
        model = ExponentialSmoothing(
            data,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=seasonal_periods,
            initialization_method="estimated",
        )
        fit_model = model.fit()
        val_name = f"HW_{self.name}"
        new_data = fit_model.fittedvalues.to_frame(name=val_name)[val_name]
        return TimeSeries(new_data, val_name)

    def nsdiffs(self, period: int, test: str = "ocsb", max_diffs: int = 3) -> int:
        if test in ["ocsb", "ch"]:
            try:
                from pdmarima.arima.utils import nsdiffs as pm_nsdiffs

                values = self.data[self.name]
                diffs = pm_nsdiffs(values, m=period, test=test, max_D=max_diffs)
                return diffs
            except ImportError:
                print(
                    "WARN: to use 'ocsb' or 'ch' tests you need to install 'pmdarima'"
                )
                print("'seas' (seasonal force) will be used instead")
                test = "seas"
        if test == "seas":
            try:
                decomp = self.decompose(model="additive", period=period)
            except:
                print("WARN: time series too short")
                return 0
            resid = decomp.resid.data
            seas = decomp.seasonal.data
            var_resid = resid.var()
            compound_var = (resid + seas).var()
            if compound_var == 0:
                return 0
            strength = max(0, 1 - (var_resid / compound_var))
            if strength > 0.64:
                return 1
            else:
                return 0
        return 0

    def decompose(
        self, model: str = "additive", period: Optional[int] = None
    ) -> DecompositionResult:
        if model == "multiplicative" and (self.data <= 0).any():
            raise ValueError(
                "Multiplicative model requires all values to be greater than zero"
            )
        res = seasonal_decompose(
            self.data, model=model, period=period, extrapolate_trend="freq"
        )
        trend_ts = TimeSeries(pd.Series(res.trend).dropna(), f"trend_{self.name}")
        seasonal_ts = TimeSeries(
            pd.Series(res.seasonal).dropna(), f"seasonal_{self.name}"
        )
        resid_ts = TimeSeries(pd.Series(res.resid).dropna(), f"resid_{self.name}")
        return DecompositionResult(self, trend_ts, seasonal_ts, resid_ts)

    def split_train_test[_Index](
        self,
        train_fraction: Optional[float] = 0.8,
        test_fraction: Optional[float] = None,
        split_index: Optional[_Index] = None,
        random_state: Optional[np.random.RandomState] = None,
    ) -> tuple['TimeSeries', 'TimeSeries']:
        raise NotImplementedError()

    def plot_acf(
        self,
        lags: int = 20,
        alpha: float = 0.05,
        return_values: bool = False,
        show_first: bool = True,
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_acf(
            self.get_values(),
            lags=lags,
            alpha=alpha,
            ax=ax,
            zero=show_first,
            title=f"Simple autocorrelation (SAC) - {self.name}",
        )
        plt.xlabel("Lags")
        plt.show()
        if return_values:
            return acf(self.data[self.name], lags)

    def plot_pacf(
        self,
        lags: int = 20,
        alpha: float = 0.05,
        method: str = "ywm",
        return_values: bool = False,
        show_first: bool = True,
    ):
        fig, ax = plt.subplots(figsize=(10, 5))
        plot_pacf(
            self.get_values(),
            lags=lags,
            alpha=alpha,
            method=method,
            ax=ax,
            zero=show_first,
            title=f"Partial autocorrelation (PAC) - {self.name}",
        )
        plt.xlabel("Lags")
        plt.show()
        if return_values:
            return pacf(self.data[self.name], lags)

    def plot_correlograms(self, lags: int = 20, show_first: bool = True):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        plot_acf(
            self.get_values(),
            lags=lags,
            zero=show_first,
            ax=ax1,
            title=f"SAC - {self.name}",
        )
        plot_pacf(
            self.get_values(),
            lags=lags,
            zero=show_first,
            method="ywm",
            ax=ax2,
            title=f"PAC - {self.name}",
        )
        plt.show()

    def plot(self, *args, **kwargs):
        plt.figure(figsize=(10, 5))
        self.data.plot(title=self.name, *args, **kwargs)
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
    print(series.apply(lambda x: 3 * x - x**2).data)

    def mi_transformacion(x):
        return 1 / np.sqrt(x)

    print(series.apply(mi_transformacion).data)

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

    # Media móvil
    print("\n- Aplicamos una media móvil -")
    print(series.moving_avg(1).data)

    # Plot rápido
    series.plot()


if __name__ == "__main__":
    _test()
