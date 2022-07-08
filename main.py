from ast import Try
from io import StringIO
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree
import streamlit as st
import pandas as pd #Open uber_pickups.py in your favorite IDE or text editor, then add these lines:
import numpy as np
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
import plotly.graph_objs  as go
import pandas as pd
from sklearn import linear_model
import streamlit.components.v1 as components
from gsheetsdb import connect
#redes neuronales
# from data_prep import features,targets, features_test, targets_test
df={}
header = []
colum = ""
yaxe = ""
def cfile():
  global df
  uploaded_file = st.file_uploader("Puede ser csv, xls, xlsx o json únicamente")
  # print("TIPO DE ARCHIVO: "+uploaded_file.type)
  if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))

    # To read file as string:
    # string_data = stringio.read()
    print(type(uploaded_file))
    te = uploaded_file.name.split(".")

    if te[-1] == 'json':
      # Can be used wherever a "file-like" object is accepted:
      df = pd.read_json(uploaded_file)
      
      df = df.to_csv()
      # st.write(dataframe)
    elif te[-1] == 'csv':
      df = pd.read_csv(uploaded_file)
    elif te[-1] == 'xlsx' or te[-1]== "xls":
      df = pd.read_excel(uploaded_file) #xlsx
      df = df.to_csv()
    else:
      st.write("Archivo incorrecto")
    st.write(df)

#! █████████████████████ ALGORITMOS █████████████████████
def regr():
  global colum, yaxe, df
  st.subheader('Graficar Regresión lineal')
  #! REGRESION LINEAL AREA DE RESULTADOS
  st.write("""La regresión lineal es una técnica de modelado estadístico que se emplea para describir una variable de respuesta continua como una función de una o varias variables predictoras. Puede ayudar a comprender y predecir el comportamiento de sistemas complejos o a analizar datos experimentales, financieros y biológicos.""")
  st.multiselect("Operaciones", [" Graficar puntos", "Definir función de tendencia", "Realizar predicción de la tendencia"])

  print(colum)
  print(yaxe)
  y = np.asarray(df[yaxe]).reshape(-1,1)
  X = np.asarray(df[colum]).reshape(-1,1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

  lrgr = linear_model.LinearRegression()
  lrgr.fit(X_train,y_train)

  preTime = st.number_input('Parametrizacion segun tiempo ingresado.')
  print(X.min(), "  ",X.max())
  preTime = st.slider('Parametrizacion segun tiempo ingresado.', int(y.min()), int(y.max()))
  st.write("PREDICCION:  ", (lrgr.predict([[preTime]])), '')

  # Generar la figura.
  if st.button('Regresión lineal'):
    # Generar la figura.1
    fig, ax = plt.subplots(figsize=(5, 3))
    regrpred = lrgr.predict(list(X))
    ax.scatter(x=df[colum], y=df[yaxe],color='red')
    plt.plot (X, regrpred, color='blue', linewidth=3)
    print(X)
    print(regrpred)
    plt.xlabel(colum)
    plt.ylabel(yaxe)
    st.pyplot(fig)
    # Generar la figura.2
    regrprednew=[]
    for i in regrpred.tolist():
      regrprednew.append(i[0])
    fig = px.scatter(x=df[colum], y=df[yaxe], opacity=1)
    fig.add_traces(go.Scatter(x =df[colum], y =  regrprednew, name='Regression Fit'))
    st.plotly_chart(fig, use_container_width=True)
    st.subheader("FUNCION DE TENDENCIA: ")
    funcion = "Y = " + str(lrgr.coef_[0]) + "*X + "+str(lrgr.intercept_)
    st.write(funcion)
    st.success('Model trained successfully')
  else:
    pass

def becdep():
  global colum, yaxe, df

def becedad():
  global colum, yaxe, df
def becuni():
  global colum, yaxe, df
def becpro():
  global colum, yaxe, df
def bechoras():
  global colum, yaxe, df
def alg():
  global colum, yaxe, df,header

  try:
    header = list(df.columns)#Extract the field names
    colum = st.selectbox("select X", header,1)
    yaxe = st.selectbox("select Y", header,1)
  except:
    pass


#! █████████████████████ OPERACIONES █████████████████████

def main():
  st.image('https://becas.usac.edu.gt/wp-content/uploads/2019/05/cropped-bannerN.png')
  st.title('Reporte de Servicio Social Becados SSE 2022')
  st.write()
  st.markdown(
    """  <style>  
    span[data-baseweb="tag"] 
    {
    background-color: blue !important;  
    }  
    </style>  """,
  unsafe_allow_html=True,
  )
  st.header('Bienvenidos :sunglasses:', anchor=None)

  cfile()
  alg()
  page_names_to_funcs = {
    "Becados por Departamento": becdep,
    "Becados por Edades": becedad,
    "Becados por Unidad Academica": becuni,
    "Becados por Proyecto en el que trabaja": becpro,
    "Total de Horas por cada Becado": bechoras
  }

  st.sidebar.image('https://www.redfia.net.gt/wp-content/uploads/2019/09/LOGO-USAC-2012-1.png')
  demo_name = st.sidebar.selectbox("Algoritmos: ", page_names_to_funcs.keys())
  page_names_to_funcs[demo_name]()
  #? Área para seleccionar las operaciones que desea realizar según lo seleccionado anteriormente.
  #? Área donde se puedan parametrizar los distintos algoritmos .
  #? Área donde se puedan visualizar de manera intuitiva los resultados. (como las gráficas).
if __name__ == "__main__":
    main()

