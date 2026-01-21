## SERIE TEMPORAL GNP (producto nacional bruto desestacionalizado)

library(haven)
gnp <- read_sav("./gnp.sav")
View(gnp)
datos <- gnp$indice

datos <- ts(datos, start = c(1947, 1), frequency = 4)
plot(datos)

#Descomponemos la serie en sus componentes y los representamos
library(timsac)
descomp=decompose(datos)
plot(descomp)

#Estabilizamos la varianza de la serie
library(forecast)
(lambda=forecast::BoxCox.lambda(datos))
datos_transf=forecast::BoxCox(datos,lambda)
plot(datos_transf)

#Calculamos el número de diferencias estacionales a realizar
library(forecast) #solo cargar si no lo hicimos previamente
(ndifes=nsdiffs(datos_transf))

# Comprobamos que, efectivamente, la serie no es estacional y 
# no necesitamos realizar ninguna diferenciación estacional.

#Calculamos el número de diferencias estacionales a realizar
library(forecast) #solo cargar si no lo hicimos previamente
(ndif=ndiffs(datos_transf))
dif_es_sim=diff(datos_transf,lag=1,differences=ndif)
plot(dif_es_sim)

# Podemos observar que en esta nueva serie, se pierde la tendencia
#Calculamos y mostramos el periodograma de la serie
#transformada y desestacionalizada
# install.packages("descomponer") #solo la primera vez
library(descomponer)
periodograma(dif_es_sim)
gperiodograma(dif_es_sim)
#Calculamos y mostramos el periodograma de la serie original
periodograma(datos_transf)
gperiodograma(datos_transf)

acf(datos_transf, lag.max=200)

#Calculamos la FAS
acf(dif_es_sim,lag.max = 20)
#Calculamos la FAP
pacf(dif_es_sim,lag.max = 20)

auto.arima(datos, allowdrift=F,trace=T)

