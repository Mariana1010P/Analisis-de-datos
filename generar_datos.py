"""
Genera datos de prueba para el proyecto. Crea un DataFrame de Pandas con los datos
generados y los almacena en un archivo CSV.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

DIARIOS = [
    "La Nación [Argentina]",
    "O Globo [Brasil]",
    "El Mercurio [Chile]",
    "El Tiempo [Colombia]",
    "La Nación [Costa Rica]",
    "La Prensa Gráfica [El Salvador]",
    "El Universal [México]",
    "El Comercio [Perú]",
    "El Nuevo Día [Puerto Rico]",
    "Listin Diario [Rep. Dominicana]",
    "El País [Uruguay]",
    "El Nacional",
]

np.random.seed(42)
dias_semanales = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

hoy = datetime.now()
fecha_inicio = hoy - timedelta(days=6 * 30)

fechas = pd.date_range(start=fecha_inicio, end=hoy)

datos = []

for diario in DIARIOS:
    for fecha in fechas:
        dia_semana = fecha.weekday()
        if dia_semana < 5:
            cantidad_articulos = np.random.randint(30, 45)
        else:
            cantidad_articulos = np.random.randint(15, 35)

        if dia_semana < 5:
            empleados_trabajando = np.random.randint(50, 80)
        else:
            empleados_trabajando = np.random.randint(20, 40)

        experiencia_promedio = np.round(np.random.uniform(3, 10), 2)

        datos.append(
            {
                "diario": diario,
                "fecha": fecha.strftime("%Y-%m-%d"),
                "dia_semana": dias_semanales[dia_semana],
                "cantidad_articulos": cantidad_articulos,
                "empleados_trabajando": empleados_trabajando,
                "experiencia_promedio": experiencia_promedio,
            }
        )

df = pd.DataFrame(datos)

df.to_csv("datos_noticias.csv", index=False)

print("Datos generados y almacenados en 'datos_noticias.csv'.")
