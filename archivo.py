import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar la data que descargamos de https://www.kaggle.com/datasets 
dataSet = pd.read_csv('C:\\Users\\FMLA GIRALDO SOTELO\\Downloads\\ASIGNATURA INTELIGENCIA ARTIFICIAL\\Actividad 2 IA\\employee_attrition_data.csv')

# Seleccionamos las características y/o variables que vamos a trabajar
caracteristicas = ['Salary', 'Satisfaction_Level', 'Average_Monthly_Hours', 'Years_at_Company', 'Promotion_Last_5Years']
X = dataSet[caracteristicas]
y = dataSet['Attrition']  # Esta es nuestra variable objetivo, es la columna que indica deserción

# Dividimos el dataset en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Creamos el modelo de Regresión Logística y usamos el parametro max_iter para aumentar las iteraciones y ayudar al algoritmo a converger
modelo = LogisticRegression(max_iter=1000)

# Entrenamos el modelo
modelo.fit(X_train, y_train)

# Hacemos predicciones
y_pred = modelo.predict(X_test)


# Calculamos la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')

