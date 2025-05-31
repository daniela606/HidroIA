import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Título de la aplicación
st.title('💧 HidroAI')
st.markdown('Aplicación inteligente para el análisis de la potabilidad del agua.')

# Carga del archivo CSV con los datos
datos = pd.read_csv('data/water_potability_sin_ceros.csv')

# Traducción de nombres de columnas al español para mejor comprensión
datos = datos.rename(columns={
    'ph': 'pH',
    'Hardness': 'Dureza',
    'Solids': 'Sólidos',
    'Chloramines': 'Cloraminas',
    'Sulfate': 'Sulfatos',
    'Conductivity': 'Conductividad',
    'Organic_carbon': 'Carbono Orgánico',
    'Trihalomethanes': 'Trihalometanos',
    'Turbidity': 'Turbidez',
    'Potability': 'Potabilidad'
})

# Muestra de los primeros registros del dataset
st.subheader('📊 Datos Iniciales')
st.write(datos.head())

# Estadísticas básicas por columna (media, desviación, etc.)
st.subheader('📈 Estadísticas Descriptivas')
st.write(datos.describe())

# Visualización del balance de clases (proporción de agua potable vs no potable)
st.subheader('⚖️ Balance de Clases (Potabilidad)')
st.bar_chart(datos['Potabilidad'].value_counts())

# Mapa de calor para visualizar la correlación entre variables
st.subheader('🧩 Mapa de Correlaciones')
figura1, eje1 = plt.subplots(figsize=(10, 8))
sns.heatmap(datos.corr(), annot=True, cmap='coolwarm', ax=eje1)
st.pyplot(figura1)

# Separación de variables independientes (entradas) y dependiente (salida)
entradas = datos.drop('Potabilidad', axis=1)
salida = datos['Potabilidad']

# Imputación de valores faltantes con la media de cada columna
entradas = entradas.fillna(entradas.mean())

# División de los datos en conjuntos de entrenamiento (70%) y prueba (30%)
entradas_entrenamiento, entradas_prueba, salida_entrenamiento, salida_prueba = train_test_split(
    entradas, salida, test_size=0.3, random_state=42)

# Creación y entrenamiento del modelo de Random Forest
modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(entradas_entrenamiento, salida_entrenamiento)

# Predicción sobre el conjunto de prueba
salida_predicha = modelo.predict(entradas_prueba)

# Cálculo y visualización de la precisión del modelo
st.subheader('📌 Precisión del Modelo')
precision = accuracy_score(salida_prueba, salida_predicha)
st.write(f'**Precisión:** {precision:.2f}')

# Generación y despliegue de un reporte completo de clasificación
st.subheader('📄 Reporte de Clasificación')
reporte = classification_report(salida_prueba, salida_predicha, output_dict=True)
reporte_df = pd.DataFrame(reporte).transpose()

# Traducción de nombres de clases y métricas al español
reporte_df = reporte_df.rename(index={
    '0': 'No Potable',
    '1': 'Potable',
    'accuracy': 'Precisión',
    'macro avg': 'Promedio Macro',
    'weighted avg': 'Promedio Ponderado'
})
# Mostrar métricas clave
st.write(reporte_df[['precision', 'recall', 'f1-score']])

# Sección de predicción manual
st.subheader('🔮 Predicción de Potabilidad del Agua')

# Entradas interactivas para el usuario
valor_ph = st.number_input('pH', min_value=0.0, max_value=14.0, value=7.0)
valor_dureza = st.number_input('Dureza', min_value=0.0, value=150.0)
valor_solidos = st.number_input('Sólidos Totales', min_value=0.0, value=10000.0)
valor_cloraminas = st.number_input('Cloraminas', min_value=0.0, value=7.0)
valor_sulfatos = st.number_input('Sulfatos', min_value=0.0, value=300.0)
valor_conductividad = st.number_input('Conductividad', min_value=0.0, value=400.0)
valor_carbono = st.number_input('Carbono Orgánico', min_value=0.0, value=10.0)
valor_trihalo = st.number_input('Trihalometanos', min_value=0.0, value=60.0)
valor_turbidez = st.number_input('Turbidez', min_value=0.0, value=3.0)

if st.button('Realizar Predicción'):
    muestra = [[
        valor_ph, valor_dureza, valor_solidos, valor_cloraminas, valor_sulfatos,
        valor_conductividad, valor_carbono, valor_trihalo, valor_turbidez
    ]]

    prediccion = modelo.predict(muestra)

    if prediccion[0] == 1:
        st.success('✅ El agua SÍ es potable.')
    else:
        st.error('❌ El agua NO es potable.')

    # Comparación visual con promedios
    st.subheader('🧪 Comparación con Promedios del Conjunto de Datos')

    entrada_usuario = pd.DataFrame([{
        'pH': valor_ph,
        'Dureza': valor_dureza,
        'Sólidos': valor_solidos,
        'Cloraminas': valor_cloraminas,
        'Sulfatos': valor_sulfatos,
        'Conductividad': valor_conductividad,
        'Carbono Orgánico': valor_carbono,
        'Trihalometanos': valor_trihalo,
        'Turbidez': valor_turbidez
    }])
    #tabla de analizis de datos
    promedio_datos = entradas.mean().to_frame().T
    promedio_datos.index = ['Promedio']
    entrada_usuario.index = ['Tu muestra']

    comparacion = pd.concat([promedio_datos, entrada_usuario])

  # Mapa de calor para comparar los valores ingresados con los promedios
    figura2, eje2 = plt.subplots(figsize=(10, 2))
    sns.heatmap(comparacion, annot=True, cmap='YlGnBu', ax=eje2)
    st.pyplot(figura2)
