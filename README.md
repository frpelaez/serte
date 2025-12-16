# serte

### Librería para trabajar con series temporales

Implementación basada en las Series de pandas. Permite construir series temporales (objetos TimeSeries) de distintas formas, incluyendo DataFrames de pandas, ficheros `.csv`, `.json` o `.xlsx` (excel). También permite realizar las operaciones elementales con series temporales (transformaciones como la logarítmica, potencial o Box-Cox o diferenciación tanto simple como estacional de cualquier orden). Incluye, además, un wrapper para plotear de manera sencilla la serie. Puede accederse a los datos de la series con el atributo `.data`, cuyo tipo es `pd.Series`, para gráficos personalizados.
