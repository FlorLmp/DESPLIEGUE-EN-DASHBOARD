#####################################################
#Importamos librerias
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st
import pandas as pd
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

#####################################################
#Diseño de pestaña
img_icon = Image.open("icon.png") #cargar la imagen 
st.set_page_config(
    page_title="Dashboard Florencia",
    page_icon= img_icon,
    layout="wide",
)

#####################################################
#Definimos la instancia
@st.cache_resource

######################################################
#Creamos la función de carga de datos
def load_data():
   #Lectura del archivo csv
   data = pd.read_csv("Florencia_limpio.csv")

   #Selecciono las columnas tipo numericas del dataframe
   numeric_data = data.select_dtypes(['float','int'])  #Devuelve Columnas
   numeric_cols= numeric_data.columns                #Devuelve lista de Columnas

   #Selecciono las columnas tipo texto del dataframe
   text_data = data.select_dtypes(['object'])  #Devuelve Columnas
   text_cols= text_data.columns              #Devuelve lista de Columnas
   
   #Selecciono algunas columnas categoricas de valores para desplegar en diferentes cuadros
   categorical_column_sex= data['room_type']
   #Obtengo los valores unicos de la columna categórica seleccionada
   unique_categories_sex= categorical_column_sex.unique()

   return data, numeric_cols, text_cols, unique_categories_sex, numeric_data

######################################################
#Cargo los datos obtenidos de la función "load_data"
data, numeric_cols, text_cols, unique_categories_sex, numeric_data = load_data()

######################################################

#Widget 1: Selectbox
#Menu desplegable de opciones de las páginas seleccionadas
View= st.selectbox(label= "View", options= ["INICIO","EXTRACCIÓN DE CARACTERÍSTICAS 📂","REGRESIÓN LINEAL SIMPLE 📈","REGRESIÓN LINEAL MULTIPLE 📊", "REGRESIÓN LOGÍSTICA 🔢"])
# CONTENIDO DE LA VISTA 1
if View == "INICIO":
    st.title("Dashboard de Florencia, Italia")
    st.write("Este dashboard presenta un Modelado explicativo usando un análisis univariado de las variables categóricas más significativas y un Modelado predictivo usando un análisis al aplicar regresión lineal simple, regresión lineal multiple y regresión logistica, esto usando datos de Airbnb acerca de la ciudad de Florencia, Italia.")
    img, title = st.columns([1, 7])
    left, right = st.columns([3, 1])
    img.image("airbnb.png", width=80)
    title.header("Acerca de Airbnb")
    left.subheader("¿Qué es?")
    left.write("Airbnb es una plataforma digital que conecta a personas que desean alquilar su propiedad con viajeros que buscan alojamiento temporal. Fundada en 2008, Airbnb ha transformado la industria del hospedaje, ofreciendo alternativas más flexibles y personalizadas que los hoteles tradicionales.A través de su modelo de economía colaborativa, permite que anfitriones publiquen espacios disponibles y que huéspedes puedan reservarlos de forma segura, utilizando filtros como precio, ubicación, tipo de propiedad, calificaciones, y más.")
    right.image("airbnbFoto.jpg", width=400)
    st.subheader("Estadisticas generales")
    st.markdown("""
                                    - Ingresos promedio por anfitrión: 30,000 euros al año
                                    - Tasa de ocupación media: 74%
                                    - Reservas anuales proemdio por propiedas: 270 noches
                                    - Mes con mayores reservar: mayo
                                    """)
    img, title = st.columns([1, 7])
    left, right = st.columns([1, 3])
    img.image("bandera.png", width=80)
    title.header("Acerca de Florencia")
    right.subheader("¿Por qué elegir Florencia?")

    right.write("Florencia es una de las joyas culturales de Italia, considerada cuna del Renacimiento y hogar de algunas de las obras de arte y arquitectura más influyentes del mundo. Su atmósfera elegante, sus calles adoquinadas llenas de historia y su importante legado artístico la hacen un destino imperdible tanto para turistas como para estudiosos del arte y la cultura.")

    right.write("En el contexto de Airbnb, Florencia representa un mercado turístico consolidado y muy competitivo, con una amplia gama de alojamientos que van desde apartamentos históricos hasta modernos espacios diseñados para viajeros. Barrios como Santa Croce, Santo Spirito o el centro histórico son zonas clave con alta demanda, ideales para analizar tendencias de hospedaje urbano.")
    left.image("FlorenciaDuomo.jpeg", width=440)
    right.image("FlorenciaPan.jpg", width=700)
    left.image("ribollita.jpeg", width=440)
    left.image("PonteVecchio.jpeg", width=440)
    st.subheader("Datos relevantes culturales")
    st.markdown("""
                                    - Nombres conocidos: La cuna del Renacimiento, la ciudad del arte, y la joya de la Toscana.
                                    -  Población: Aproximadamente 366,000 habitantes.
                                    -  Principales lugares turísticos:
                                        - Catedral de Santa Maria del Fiore (Duomo)
                                        - Galleria degli Uffizi
                                        - Ponte Vecchio
                                        - Palazzo Pitti
                                        - Piazza della signoria
                                    -  Gastronomía icónica:
                                        - Bistecca alla fiorentina
                                        - Ribollita
                                        - Pappa al pomodoro
                                    """)

###############################################################################
# CONTENIDO DE LA VISTA 2
elif View == "EXTRACCIÓN DE CARACTERÍSTICAS 📂":
    #Generamos los encabezados para el dashboard
    st.title("Extracción de Características")
    st.header("Panel Principal")
    st.write(data)
    #######################
    ####Generamoos sidebar
    st.sidebar.title("Extracción de Características")
    st.sidebar.header("Sidebar")

    ###############################
    #widget 2: Checkbok
##condicional para que aparezca el dataframe

###############################################################################
# CONTENIDO DE LA VISTA 3
elif View == "REGRESIÓN LINEAL SIMPLE 📈":
#Generamos los encabezados para el dashboard
    st.title("REGRESIÓN LINEAL SIMPLE")
    st.title("Modelado Predictivo")
    
    # Sidebar para selección de variables
    st.sidebar.title("Configuración del Modelo")
    numeric_cols = [col for col in numeric_cols if data[col].nunique() > 1]  # Filtra variables constantes
    
    # Selectores de variables
    feature = st.sidebar.selectbox("Variable Independiente (X)", numeric_cols)
    target = st.sidebar.selectbox("Variable Dependiente (y)", numeric_cols)
    
    # División de datos
    X = data[[feature]]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Modelado
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Métricas
    st.subheader("Resultados del Modelo")
    st.write(f"Coeficiente: {model.coef_[0]:.4f}")
    st.write(f"Intercepto: {model.intercept_:.4f}")
    st.write(f"R²: {model.score(X_test, y_test):.4f}")
    
    # Gráfico
    fig = px.scatter(x=X_test[feature], y=y_test, opacity=0.7, 
                     labels={'x': feature, 'y': target},
                     title=f"Regresión: {target} ~ {feature}")
    fig.add_scatter(x=X_test[feature], y=y_pred, mode='lines', name='Línea de Regresión')
    st.plotly_chart(fig, use_container_width=True)
    
    
    
# CONTENIDO DE LA VISTA 4
elif View == "REGRESIÓN LINEAL MULTIPLE 📊":
#Generamos los encabezados para el dashboard
    st.title("REGRESIÓN LINEAL MULTIPLE")
    st.title("Modelado Predictivo")
    st.header("Regresión Lineal Múltiple")
    
    st.sidebar.title("Configuración del Modelo")
    selected_features = st.sidebar.multiselect("Variables Independientes (X)", numeric_cols)
    target = st.sidebar.selectbox("Variable Dependiente (y)", numeric_cols)
    
    if len(selected_features) >= 2:
        X = data[selected_features]
        y = data[target]
        
        # Estandarización
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        st.subheader("Resultados del Modelo")
        st.write("Coeficientes:", dict(zip(selected_features, model.coef_)))
        st.write(f"Intercepto: {model.intercept_:.4f}")
        st.write(f"R²: {model.score(X_test, y_test):.4f}")
        
        # Gráfico de residuos
        residuals = y_test - y_pred
        fig = px.scatter(x=y_pred, y=residuals,
                         labels={'x': 'Predicciones', 'y': 'Residuos'},
                         title="Análisis de Residuos")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Selecciona al menos 2 variables predictoras")

# CONTENIDO DE LA VISTA 4
elif View == "REGRESIÓN LOGÍSTICA 🔢":
#Generamos los encabezados para el dashboard
    st.title("REGRESIÓN LOGÍSTICA")
    # Verificar si hay variable binaria
    binary_vars = [col for col in numeric_cols if data[col].nunique() == 2]
    
    if not binary_vars:
        st.warning("No se encontraron variables binarias para regresión logística.")
        if st.sidebar.checkbox("Crear variable binaria artificial"):
            selected_num = st.sidebar.selectbox("Selecciona variable numérica para binarizar", numeric_cols)
            median_val = data[selected_num].median()
            data['target'] = (data[selected_num] > median_val).astype(int)
            binary_vars = ['target']
    
    if binary_vars:
        # Selección de variables
        col1, col2 = st.columns(2)
        with col1:
            y_var = st.sidebar.selectbox("Variable Objetivo (Y)", binary_vars)
        with col2:
            x_var = st.sidebar.selectbox("Variable Predictora (X)", numeric_cols)
        
        # Gráfico de dispersión con regresión logística
        fig = px.scatter(data, x=x_var, y=y_var, 
                        title="Regresión Logística")
        st.plotly_chart(fig, use_container_width=True)
        
        # Ajustar modelo
        X = data[[x_var]].values
        y = data[y_var].values
        
        # Estandarizar datos
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        # Métricas
        from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
        
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Mostrar métricas
        st.subheader("Métricas del Modelo")
        colE, colS, colP, colF = st.columns(4)
        colE.metric("Exactitud", f"{accuracy*100:.2f}%")
        colS.metric("Sensibilidad", f"{recall*100:.2f}%")
        colP.metric("Precisión", f"{precision*100:.2f}%")
        colF.metric("F1-Score", f"{f1*100:.2f}%")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicho 0', 'Predicho 1'],
            y=['Real 0', 'Real 1'],
            text=[[f"VN: {cm[0,0]}", f"FP: {cm[0,1]}"], 
                 [f"FN: {cm[1,0]}", f"VP: {cm[1,1]}"]],
            texttemplate="%{text}",
            colorscale='Blues',
            showscale=False
        ))
        
        fig.update_layout(
            title="Matriz de Confusión",
            xaxis_title="Predicción",
            yaxis_title="Valor Real"
        )
        
        st.plotly_chart(fig, use_container_width=True)
