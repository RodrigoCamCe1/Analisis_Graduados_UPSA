import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo CSV con una cadena sin formato
df = pd.read_csv(r"C:\Users\intoy\OneDrive\Documentos\Tareas\Gestión-de-Bases-de-Datos\GRADUADOS2.csv", sep=";")

# Mostrar las primeras filas para verificar que los datos se cargaron correctamente
print(df.head())


# Evolución de graduados por año
graduados_por_año = df['Año'].value_counts().sort_index()

plt.figure(figsize=(10, 6))
sns.barplot(x=graduados_por_año.index, y=graduados_por_año.values, palette="viridis")
plt.title('Evolución de Graduados por Año')
plt.xlabel('Año')
plt.ylabel('Cantidad de Graduados')
plt.xticks(rotation=45)
plt.show()


# Evolución de graduados por facultad
graduados_por_facultad = df['Facultad'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=graduados_por_facultad.index, y=graduados_por_facultad.values, palette="plasma")
plt.title('Evolución de Graduados por Facultad')
plt.xlabel('Facultad')
plt.ylabel('Cantidad de Graduados')
plt.xticks(rotation=45)
plt.show()


# Evolución de graduados por carrera
graduados_por_carrera = df['Carrera'].value_counts()

plt.figure(figsize=(12, 6))
sns.barplot(x=graduados_por_carrera.index, y=graduados_por_carrera.values, palette="magma")
plt.title('Evolución de Graduados por Carrera')
plt.xlabel('Carrera')
plt.ylabel('Cantidad de Graduados')
plt.xticks(rotation=45)
plt.show()


# Análisis del comportamiento a través del tiempo (Tendencia de Graduados)
graduados_tendencia = df.groupby(['Año', 'Carrera']).size().reset_index(name='Cantidad')

plt.figure(figsize=(12, 6))
sns.lineplot(data=graduados_tendencia, x='Año', y='Cantidad', hue='Carrera', marker='o')
plt.title('Tendencia de Graduados por Año y Carrera')
plt.xlabel('Año')
plt.ylabel('Cantidad de Graduados')
plt.legend(title="Carrera", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()


from sklearn.linear_model import LinearRegression

# Filtrar los datos de graduados por año
data = df.groupby('Año').size().reset_index(name='Cantidad')

# Preparar los datos para el modelo de regresión lineal
X = data['Año'].values.reshape(-1, 1)
y = data['Cantidad'].values

# Crear el modelo de regresión
model = LinearRegression()
model.fit(X, y)

# Realizar la predicción para los próximos años
predicciones = model.predict([[2025], [2026], [2027]])

print(f"Predicción de graduados para 2025: {predicciones[0]}")
print(f"Predicción de graduados para 2026: {predicciones[1]}")
print(f"Predicción de graduados para 2027: {predicciones[2]}")


# Estadísticas generales por carrera
estadisticas_carrera = df.groupby('Carrera').size().describe()
print("Estadísticas generales de graduados por carrera:")
print(estadisticas_carrera)

# Estadísticas generales por facultad
estadisticas_facultad = df.groupby('Facultad').size().describe()
print("\nEstadísticas generales de graduados por facultad:")
print(estadisticas_facultad)

# Estadísticas generales de graduados (general)
estadisticas_general = df['Año'].value_counts().describe()
print("\nEstadísticas generales de graduados (general):")
print(estadisticas_general)



