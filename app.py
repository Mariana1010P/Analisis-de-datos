import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class NewsApp:
    """
    Clase para la interfaz de usuario de gestión de artículos de noticias.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Gestor de Artículos de Noticias")

        self.data = pd.DataFrame()

        self.setup_ui()
        self.load_data()

    def setup_ui(self):
        """Configuración de la interfaz de usuario."""
        frame = tk.Frame(self.root)
        frame.pack(pady=20)

        self.diario_combobox = ttk.Combobox(
            frame,
            values=[
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
            ],
        )
        self.diario_combobox.grid(row=0, column=0)

        self.cantidad_entry = tk.Entry(frame)
        self.cantidad_entry.grid(row=0, column=1)

        self.add_button = tk.Button(
            frame, text="Agregar Artículo", command=self.add_article
        )
        self.add_button.grid(row=0, column=2)

        # Tabla para mostrar datos
        self.tree = ttk.Treeview(
            self.root, columns=("Diario", "Fecha", "Cantidad"), show="headings"
        )
        self.tree.heading("Diario", text="Diario")
        self.tree.heading("Fecha", text="Fecha")
        self.tree.heading("Cantidad", text="Cantidad")
        self.tree.pack(pady=20)

        # Botón para mostrar el reporte de la semana pasada
        self.report_button = tk.Button(
            self.root, text="Mostrar Reporte Semana Pasada", command=self.show_report
        )
        self.report_button.pack(pady=20)

        # Botón para predecir artículos para mañana
        self.predict_button = tk.Button(
            self.root,
            text="Predecir Artículos para Mañana",
            command=self.predict_articles,
        )
        self.predict_button.pack(pady=20)

    def load_data(self):
        """Carga los datos del archivo CSV."""
        try:
            self.data = pd.read_csv("datos_noticias.csv")
            # Asegurarse de que las columnas de fecha son del tipo correcto
            self.data["fecha"] = pd.to_datetime(self.data["fecha"], errors="coerce")
            self.data.dropna(
                subset=["fecha"], inplace=True
            )  # Eliminar filas con fechas inválidas
            print(self.data)  # Verifica que se cargaron los datos correctamente
            self.update_table()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")

    def update_table(self):
        """Actualiza la tabla de datos."""
        for row in self.tree.get_children():
            self.tree.delete(row)

        if not self.data.empty:  # Verifica que hay datos
            for _, row in self.data.iterrows():
                self.tree.insert(
                    "",
                    "end",
                    values=(
                        row["diario"],
                        row["fecha"].strftime("%Y-%m-%d"),
                        row["cantidad_articulos"],
                    ),
                )
        else:
            print("No hay datos para mostrar.")

    def calculate_coefficient_of_variation(self, data):
        """Calcula el coeficiente de variación."""
        mean = data.mean()
        std_dev = data.std()
        if mean == 0:  # Para evitar división por cero
            return 0
        return (std_dev / mean) * 100  # Expresado como porcentaje

    def check_article_count(self, diario, cantidad):
        """Verifica la cantidad de artículos cargada."""
        today = datetime.now().date()

        # Filtrar datos de los últimos 6 meses
        six_months_ago = today - timedelta(days=180)

        # Obtener datos históricos para el mismo día de la semana y diario
        weekly_data = self.data[
            (
                self.data["fecha"] >= six_months_ago.strftime("%Y-%m-%d")
            )  # Datos de los últimos 6 meses
            & (
                self.data["dia_semana"] == today.strftime("%A")
            )  # Mismo día de la semana
            & (self.data["diario"] == diario)  # Mismo diario
        ]

        # 1ra fase: Calcular el promedio en decimal y compararlo con la cantidad ingresada
        if weekly_data.empty:
            return False, "No hay datos históricos suficientes para validar."

        avg = weekly_data["cantidad_articulos"].mean()  # Promedio en decimal
        print(f"Promedio para el dia {today} del diario {diario} es: {round(avg,2)}")
        threshold = avg - 8  # Ajusta el umbral en función del promedio
        if cantidad < round(threshold):
            return (
                False,
                f"La cantidad cargada ({cantidad}) está por debajo del umbral esperado ({round(threshold)}).",
            )

        # 2da fase: Calcular el coeficiente de variación
        cv = self.calculate_coefficient_of_variation(weekly_data["cantidad_articulos"])
        print(
            f"Coeficiente de variación para el día {today} del diario {diario} es: {cv}"
        )

        if cv > 30:  # Alta variabilidad
            # Revisar si la cantidad está dentro del rango intercuartil (IQR)
            q1 = weekly_data["cantidad_articulos"].quantile(0.25)
            q3 = weekly_data["cantidad_articulos"].quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr

            if cantidad < lower_bound or cantidad > upper_bound:
                return (
                    False,
                    f"La cantidad cargada ({cantidad}) está fuera del rango intercuartil ({round(lower_bound, 2)}, {round(upper_bound, 2)}).",
                )
        else:  # Baja variabilidad
            # 3ra fase: Revisar la frecuencia de la cantidad de artículos
            frequency_data = weekly_data["cantidad_articulos"].value_counts()
            most_frequent_quantity = frequency_data.idxmax()  # Cantidad más frecuente

            if cantidad != most_frequent_quantity:
                return (
                    False,
                    f"La cantidad cargada ({cantidad}) no coincide con la cantidad más frecuente ({most_frequent_quantity}).",
                )

        return True, "Artículo validado exitosamente."

    def train_regression_model(self, diario):
        """Entrena el modelo de regresión lineal con los datos del diario especificado."""
        try:
            # Filtrar datos por diario
            diario_data = self.data[self.data["diario"] == diario]

            # Verificar si existen las columnas necesarias en los datos
            if all(
                col in diario_data.columns
                for col in [
                    "cantidad_articulos",
                    "empleados_trabajando",
                    "experiencia_promedio",
                ]
            ):
                x = diario_data[["empleados_trabajando", "experiencia_promedio"]]
                y = diario_data["cantidad_articulos"]

                # Asegurarse de que existen suficientes datos
                if len(diario_data) < 5:  # Establecer un umbral mínimo
                    messagebox.showerror(
                        "Error", "No hay suficientes datos para entrenar el modelo."
                    )
                    return None

                # Dividir los datos en conjunto de entrenamiento y prueba
                x_train, x_test, y_train, y_test = train_test_split(
                    x, y, test_size=0.2, random_state=42
                )

                # Crear y entrenar el modelo de regresión lineal
                model = LinearRegression()
                model.fit(x_train, y_train)

                # Hacer predicciones
                y_pred = model.predict(x_test)

                # Evaluar el modelo
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                print(
                    f"[{diario}] Mean Squared Error: {mse:.2f}"
                )  # Imprimir resultados
                print(f"[{diario}] R^2 Score: {r2:.2f}")  # Imprimir resultados

                return model  # Retornar el modelo entrenado
            else:
                messagebox.showerror(
                    "Error", "Faltan columnas necesarias en los datos."
                )
                return None
        except Exception as e:
            messagebox.showerror("Error", f"Error al entrenar el modelo: {e}")
            return None

    def predict_articles(self):
        """Predice la cantidad de artículos que se subirán mañana para cada diario."""
        try:
            if self.data.empty:
                messagebox.showerror(
                    "Error", "No hay datos suficientes para realizar la predicción."
                )
                return

            predicciones = []

            # Predecir para cada diario
            for diario in self.diario_combobox["values"]:
                # Filtrar datos por diario y últimos 6 meses
                today = datetime.now()
                six_months_ago = today - timedelta(days=180)
                diario_data = self.data[
                    (self.data["diario"] == diario)
                    & (self.data["fecha"] >= six_months_ago)
                ]

                # Si hay datos para el diario, proceder
                if not diario_data.empty:
                    model = self.train_regression_model(diario)

                    if model:
                        # Usar los datos más recientes para la predicción
                        last_data = diario_data.sort_values(by="fecha").iloc[-1]

                        # Crear los datos de entrada para la predicción basados en los valores más recientes
                        input_data = np.array(
                            [
                                [
                                    last_data["empleados_trabajando"],
                                    last_data["experiencia_promedio"],
                                ]
                            ]
                        )

                        # Realizar la predicción
                        prediction = model.predict(input_data)[0]
                        predicciones.append((diario, int(prediction)))

            # Mostrar resultados en una nueva ventana
            prediction_window = tk.Toplevel(self.root)
            prediction_window.title("Predicciones de Artículos para Mañana")

            # Crear una tabla para mostrar las predicciones
            prediction_tree = ttk.Treeview(
                prediction_window, columns=("Diario", "Predicción"), show="headings"
            )
            prediction_tree.heading("Diario", text="Diario")
            prediction_tree.heading("Predicción", text="Predicción")
            prediction_tree.pack(pady=20)

            # Insertar las predicciones en la tabla
            for diario, pred in predicciones:
                prediction_tree.insert("", "end", values=(diario, pred))

        except Exception as e:
            messagebox.showerror("Error", f"Error al realizar la predicción: {e}")

    def add_article(self):
        """Agrega un nuevo artículo y verifica la cantidad cargada."""
        diario = self.diario_combobox.get()
        cantidad_str = self.cantidad_entry.get()

        if not diario or not cantidad_str:
            messagebox.showwarning(
                "Advertencia", "Por favor, completa todos los campos."
            )
            return
        
        try:
            # Convertir la cantidad a entero
            cantidad = int(cantidad_str)

            # Verificar que la cantidad no sea negativa
            if cantidad < 0:
                messagebox.showwarning("Advertencia", "La cantidad debe ser un número positivo.")
                return

            fecha = datetime.now()

            # Crear un nuevo DataFrame con los datos del nuevo artículo
            new_data = pd.DataFrame(
                {
                    "diario": [diario],
                    "fecha": [fecha],
                    "cantidad_articulos": [cantidad],
                    "dia_semana": [fecha.strftime("%A")],
                    "empleados_trabajando": [
                        np.random.randint(10, 50)
                    ],  # Datos aleatorios para la demo
                    "experiencia_promedio": [
                        np.random.uniform(1, 10)
                    ],  # Datos aleatorios para la demo
                }
            )

            # Usar pd.concat() para agregar el nuevo DataFrame
            self.data = pd.concat([self.data, new_data], ignore_index=True)

            # Verificar la cantidad cargada
            is_valid, message = self.check_article_count(diario, cantidad)

            # Mostrar mensaje basado en la validación
            if not is_valid:
                messagebox.showwarning(
                    "Alerta", message
                )  # Muestra la alerta con el motivo
            else:
                # Guardar datos en CSV
                self.data.to_csv("datos_noticias.csv", index=False)
                messagebox.showinfo("Éxito", "Artículo guardado exitosamente.")

            self.update_table()
            self.cantidad_entry.delete(0, tk.END)
        except ValueError:
            messagebox.showerror("Error", "La cantidad debe ser un número entero.")
        except Exception as e:
            messagebox.showerror(
                "Error", f"Ocurrió un error al agregar el artículo: {e}"
            )

    def show_report(self):
        """
        Muestra el reporte de artículos por diario en la consola y en una nueva ventana.
        """
        today = datetime.now().date()
        last_week_data = self.data[
            self.data["fecha"] >= (today - timedelta(days=7)).strftime("%Y-%m-%d")
        ]

        if last_week_data.empty:
            messagebox.showinfo(
                "Reporte de Artículos",
                "No hay datos disponibles para la última semana.",
            )
            return

        report_data = {}
        for _, row in last_week_data.iterrows():
            diario = row["diario"]
            cantidad = row["cantidad_articulos"]
            dia_semana = row["dia_semana"]
            if diario not in report_data:
                report_data[diario] = {
                    day: 0
                    for day in [
                        "Monday",
                        "Tuesday",
                        "Wednesday",
                        "Thursday",
                        "Friday",
                        "Saturday",
                        "Sunday",
                    ]
                }
            report_data[diario][dia_semana] += cantidad

        report_table = []
        for diario, dias in report_data.items():
            total = sum(dias.values())
            # Promedio solo si hay datos para evitar división por cero
            avg = (
                total / len([day for day in dias.values() if day > 0])
                if any(dias.values())
                else 0
            )
            report_table.append(
                [diario] + list(dias.values()) + [round(avg), 2]
            )  # Cambiado a int

        # Mostrar reporte en un nuevo Toplevel
        report_window = tk.Toplevel(self.root)
        report_window.title("Reporte Semanal")

        columns = ["Diario"] + [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
            "Promedio",
        ]
        report_tree = ttk.Treeview(report_window, columns=columns, show="headings")
        for col in columns:
            report_tree.heading(col, text=col)
        report_tree.pack(fill="both", expand=True)

        for row in report_table:
            report_tree.insert("", "end", values=row)

        report_tree.pack()


if __name__ == "__main__":
    root = tk.Tk()
    app = NewsApp(root)
    root.mainloop()
