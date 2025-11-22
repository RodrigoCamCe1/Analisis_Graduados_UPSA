import pandas as pd

# Cargar el archivo CSV
df = pd.read_csv(r"C:\Users\intoy\OneDrive\Documentos\Tareas\Gestión-de-Bases-de-Datos\GRADUADOS2.csv", sep=";", header=None)

# Mostrar el número de columnas
print(f'Número de columnas en el archivo CSV: {df.shape[1]}')

# Mostrar las primeras filas para verificar las columnas
print(df.head())
