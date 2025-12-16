import numpy as np

from core import TimeSeries


def main() -> None:
    # Leer de csv (también se puede de pd.DataFrame o de excel)
    path = "data/data.csv"
    series = TimeSeries.from_csv(path, value_col_name="sells", index_col_name="dates")

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
    main()
