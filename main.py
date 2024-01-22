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
# Create a connection object.
conn = connect()
# Perform SQL query on the Google Sheet.
# Uses st.cache to only rerun when the query changes or after 10 min.

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


@st.cache(ttl=600)
def run_query(query):
  rows = conn.execute(query, headers=1)
  rows = rows.fetchall()
  return rows
#! JALANDO LAS FILAS DE CADA HOJA.
reportes = st.secrets["reportes"]
rowsreportes = run_query(f'SELECT * FROM "{reportes}"')
carne = [] #* CARNES
for row in rowsreportes:
  carne.append(row[0])

NamesAll = [] #* TODOS NOMBRES
for row in rowsreportes:
  NamesAll.append(row[1])
resformularios  = st.secrets["resformularios"]
rowsresformularios = run_query(f'SELECT * FROM "{resformularios}"')
# st.write("TIPO DE DATO DE LAS LISTAS: ", type(rowsresformularios))

#! ORGANNIZE DATA
BecNames = [] #* NOMBRE DE LOS BECADOS
for row in rowsresformularios:
    BecNames.append(row[1])

BecHoras = [] #* HORAS DE LOS BECADOS
for row in rowsresformularios:
    BecHoras.append(row[2])

BecLugar = [] #* LUGAR DE LOS BECADOS
for row in rowsresformularios:
    BecLugar.append(row[5])

BecTipo = [] #* TIPO DE TRABAJO
for row in rowsresformularios:
    BecTipo.append(row[7])

BecEdad = [] #* EDAD DEL BECADO
for row in rowsresformularios:
    if row[8] is not None:
      BecEdad.append(int(row[8]))

Becdeptoreplic = [] #* DEPTO DE REPLICA
for row in rowsresformularios:
    Becdeptoreplic.append(row[9])

BecCapacitadas = [] #* PERSONAS CAPACITADAS
for row in rowsresformularios:
    BecCapacitadas.append(row[10])
# ! 2022
def becdep():
  global rowsreportes, rowsresformularios
  # st.write(rowsresformularios)
  # *La división política de Guatemala consta de 22 Departamentos:
    # 1  Alta Verapaz
    # 2 Baja Verapaz
    # 3 Chimaltenango
    # 4 Chiquimula
    # 5 El Progreso
    # 6 Escuintla
    # 7 Guatemala
    # 8 Huehuetenango
    # 9 Izabal
    # 10 Jalapa
    # 11 Jutiapa
    # 12 Petén
    # 13 Quetzaltenango
    # 14 Quiché
    # 15 Retalhuleu
    # 16 Sacatepéquez
    # 17 San marcos
    # 18 Santa rosa
    # 19 Sololá
    # 20 Suchitepéquez
    # 21 Totonicapán
    # 22 Zacapa
  dep=[
  [15.5, -90.333333],
  [15.1009234, -90.3139743],
  [14.6622, -90.8208],
  [14.8, -89.54583],
  [14.3579867773633, -89.84790854555509],
  [14.3009, -90.78581],
  [14.64072, -90.51327],
  [15.31918, -91.47241],
  [15.47225, -88.8407],
  [14.63472, -89.98889],
  [14.29167, -89.89583],
  [16.8, -89.93333],
  [14.83472, -91.51806],
  [15.03085, -91.14871],
  [14.53611, -91.67778],
  [14.578414124412356, -90.79401954555287],
  [14.96389 , -91.79444],
  [14.15015235, -90.3508818353375],
  [14.77222 , -91.18333],
  [14.37766785, -91.3643907717613],
  [14.91167 , -91.36111],
  [14.97222, -89.53056]]
  depto = []
  na = st.selectbox("Nombre del Estudiante: ", BecNames,1)
  no=0
  for w in BecNames:
    if w == na:
      break
    no+=1
  if na:
# tx= "Estudiante:  "+ na+ "\n pertenece al departamento de: \n "+BecLugar[no] 
    # st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>', unsafe_allow_html=True)
    # st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">pertenece al departamento de: </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecLugar[no]}</p>', unsafe_allow_html=True)


  for B in BecLugar:
    if B == 'Alta Verapaz':
      depto.append(dep[0])
    elif B == 'Baja Verapaz':
      depto.append(dep[1])
    elif B == 'Chimaltenago':
      depto.append(dep[2])
    elif B == 'Chiquimula':
      depto.append(dep[3])
    elif B == 'El Progreso':
      depto.append(dep[4])
    elif B == 'Escuintla':
      depto.append(dep[5])
    elif B == 'Guatemala':
      depto.append(dep[6])
    elif B == 'Huehuetenango':
      depto.append(dep[7])
    elif B == 'Izabal':
      depto.append(dep[8])
    elif B == 'Jalapa':
      depto.append(dep[9])
    elif B == 'Jutiapa':
      depto.append(dep[10])
    elif B == 'Petén':
      depto.append(dep[11])
    elif B == 'Quetzaltenango':
      depto.append(dep[12])
    elif B == 'Quiché':
      depto.append(dep[13])
    elif B == 'Retalhuleu':
      depto.append(dep[14])
    elif B == 'Sacatepéquez':
      depto.append(dep[15])
    elif B == 'San Marcos':
      depto.append(dep[16])
    elif B == 'Santa Rosa':
      depto.append(dep[17])
    elif B == 'Sololá':
      depto.append(dep[18])
    elif B == 'Suchitepéquez':
      depto.append(dep[19])
    elif B == 'Totonicapán':
      depto.append(dep[20])
    elif B == 'Zacapa':
      depto.append(dep[21])

  data = pd.DataFrame(
    np.array(depto),
    # ['Alta Verapaz', 'Baja Verapaz', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango'],
    columns=['lat', 'lon'])
  st.map(data)

  
  # latlist = []
  # for d in depto:
  #   latlist.append(d[0])
  # lonlist = []
  # for d in depto:
  #   lonlist.append(d[1])
  # data = pd.DataFrame(
  #   BecNames,
  #   latlist,
  #   lonlist)
  # st.write(data)
  # fig = px.scatter_mapbox(data, lat=latlist, lon=lonlist,size_max=15, zoom=10)
  # st.plotly_chart(fig)

def becedad():
  global rowsreportes, rowsresformularios
  na = st.selectbox("Nombre del Estudiante: ", BecNames, 1)
  no = 0
  for w in BecNames:
      if w == na:
          break
      no += 1
  if na:
      # st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>',
      #             unsafe_allow_html=True)
      # st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>',
      #             unsafe_allow_html=True)
      st.markdown(
          f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Tiene: </p>',
          unsafe_allow_html=True)
      st.markdown(
          f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecEdad[no]} Años.</p>',
          unsafe_allow_html=True)

  st.subheader("Gráfico de Edades (Barras)")
  chart_data = pd.DataFrame({'Nombres': BecNames, 'Edades': BecEdad})
  st.bar_chart(chart_data.set_index('Nombres'))

  # st.subheader("VISTA 2 EDAD DE BECADOS (Sunburst)")
  # acumconts = [str(int(carne[NamesAll.index(w)])) for w in BecNames]
  # acumcontstemp = acumconts
  # acumconts.pop()
  # data = dict(
  #     names=acumcontstemp,
  #     parent=acumconts.insert(0, ""),
  #     value=BecEdad)

  # fig = px.sunburst(
  #     data,
  #     names='names',
  #     parents='parent',
  #     values='value',
  # )
  # st.plotly_chart(fig)

def becuni():
  global rowsreportes, rowsresformularios
  hist_data = [x1, x2, x3]
  group_labels = ['Group 1', 'Group 2', 'Group 3']
  fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
  st.plotly_chart(fig, use_container_width=True)

def becpro():
  global rowsreportes, rowsresformularios

  # Obtener el nombre seleccionado
  na = st.selectbox("Nombre del Estudiante: ", BecNames, 1)
  no = 0

  # Encontrar el índice del estudiante seleccionado
  for w in BecNames:
      if w == na:
          break
      no += 1

  # Mostrar información del estudiante seleccionado
  if na:
      st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Esta trabajando en los ejes de: </p>', unsafe_allow_html=True)
      st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecTipo[no]} </p>', unsafe_allow_html=True)

  # Crear una lista de listas para los datos de la tabla
  ndata = [['BECADOS', 'AREA DE TRABAJO', 'DEPTO DONDE TRABAJA']]

  # Llenar la lista con datos
  for b in BecNames:
      ndata.append([b, BecTipo[BecNames.index(b)], Becdeptoreplic[BecNames.index(b)]])

  # Crear un DataFrame para la tabla
  tabla_datos = pd.DataFrame(ndata[1:], columns=ndata[0])

  # Mostrar la tabla en Streamlit
  st.subheader("Área de Trabajo y Departamento de Todos los Estudiantes")
  st.table(tabla_datos)


def bechoras():
  global rowsreportes, rowsresformularios
  na = st.selectbox("Nombre del Estudiante: ", BecNames,1)
  no=0
  for w in BecNames:
    if w == na:
      break
    no+=1
  if na:
    # st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>', unsafe_allow_html=True)
    # st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">tiene Acumuladas: </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecHoras[no]} Horas.</p>', unsafe_allow_html=True)

  st.subheader("HORAS POR BECADOS")
  # wide_df = px.data.medals_wide()
  # st.write(px.data.medals_wide())
  # st.write(type(px.data.medals_wide()))
  acumconts=[]
  for w in BecNames:
    acumconts.append(str(int(carne[NamesAll.index(w)])))
  wide_df = {tuple(BecNames),tuple(BecHoras)}
  fig = px.bar(wide_df, x=acumconts, y=BecHoras,color=BecNames, title="HORAS REALIZADAS POR CADA BECADO (GRAFICO)")
  st.plotly_chart(fig)

  st.subheader("VISTA 2")
  acumcontstemp=acumconts
  acumconts.pop()
  data = dict(
    names=acumcontstemp,
    parent=acumconts.insert(0,""),
    value=BecHoras)

  fig = px.sunburst(
      data,
      names='names',
      parents='parent',
      values='value',
  )
  st.plotly_chart(fig)

def becbeneficiadas():
  global rowsreportes, rowsresformularios
  # Crear un diccionario para almacenar el total de personas beneficiadas por departamento
  total_personas_por_depto = {}

  for i in range(len(BecNames)):
      becado_departamento = BecLugar[i]
      personas_capacitadas = BecCapacitadas[i]

      # Verificar si hay un valor numérico para personas capacitadas
      if isinstance(personas_capacitadas, (int, float)):
          # Sumar el número de personas capacitadas al total del departamento
          if becado_departamento in total_personas_por_depto:
              total_personas_por_depto[becado_departamento] += personas_capacitadas
          else:
              total_personas_por_depto[becado_departamento] = personas_capacitadas

  # Mostrar el total de personas beneficiadas por departamento
  st.subheader("Total de Personas Beneficiadas por Departamento")
  # for depto, total_personas in total_personas_por_depto.items():
  #     st.write(f"{depto}: {int(total_personas)} personas beneficiadas")

  # Crear un gráfico de barras para visualizar los totales por departamento

  # Convertir el diccionario a un DataFrame
  chart_data = pd.DataFrame(list(total_personas_por_depto.items()), columns=['Departamento', 'Total Personas'])

  # Crear una gráfica de barras con Plotly Express
  fig = px.bar(chart_data, x='Departamento', y='Total Personas', title='Total de Personas Beneficiadas por Departamento')

  # Mostrar la gráfica en Streamlit
  st.plotly_chart(fig)

    # Mostrar el total de personas beneficiadas por departamento en una tabla
  st.subheader("Total de Personas Beneficiadas por Departamento")

  # Crear un DataFrame con los datos
  tabla_datos = pd.DataFrame(list(total_personas_por_depto.items()), columns=['Departamento', 'Total Personas Beneficiadas'])

  # Formatear los números en el DataFrame
  tabla_datos['Total Personas Beneficiadas'] = tabla_datos['Total Personas Beneficiadas'].map('{:.0f}'.format)

  # Mostrar la tabla en Streamlit
  st.table(tabla_datos)

#! JALANDO LAS FILAS DE CADA HOJA 2023.
reportes2 = st.secrets["reportes2"]
rowsreportes2 = run_query(f'SELECT * FROM "{reportes2}"')
carne2 = [] #* CARNES
for row in rowsreportes2:
  carne2.append(row[0])

NamesAll2 = [] #* TODOS NOMBRES
for row in rowsreportes2:
  NamesAll2.append(row[1])
resformularios2  = st.secrets["resformularios2"]
rowsresformularios2 = run_query(f'SELECT * FROM "{resformularios2}"')
# st.write("TIPO DE DATO DE LAS LISTAS: ", type(rowsresformularios))

#! ORGANNIZE DATA
BecNames2 = [] #* NOMBRE DE LOS BECADOS
for row in rowsresformularios2:
    BecNames2.append(row[1])

BecHoras2 = [] #* HORAS DE LOS BECADOS
for row in rowsresformularios2:
    BecHoras2.append(row[2])

BecLugar2 = [] #* LUGAR DE LOS BECADOS
for row in rowsresformularios2:
    BecLugar2.append(row[5])

BecTipo2 = [] #* TIPO DE TRABAJO
for row in rowsresformularios2:
    BecTipo2.append(row[7])

BecEdad2 = [] #* EDAD DEL BECADO
for row in rowsresformularios2:
    if row[8] is not None:
      BecEdad2.append(int(row[8]))

Becdeptoreplic2 = [] #* DEPTO DE REPLICA
for row in rowsresformularios2:
    Becdeptoreplic2.append(row[9])

BecCapacitadas2 = [] #* PERSONAS CAPACITADAS
for row in rowsresformularios2:
    BecCapacitadas2.append(row[10])
# ! 2023
def becdep2():
  global rowsreportes2, rowsresformularios2
  # st.write(rowsresformularios2)
  # *La división política de Guatemala consta de 22 Departamentos:
    # 1  Alta Verapaz
    # 2 Baja Verapaz
    # 3 Chimaltenango
    # 4 Chiquimula
    # 5 El Progreso
    # 6 Escuintla
    # 7 Guatemala
    # 8 Huehuetenango
    # 9 Izabal
    # 10 Jalapa
    # 11 Jutiapa
    # 12 Petén
    # 13 Quetzaltenango
    # 14 Quiché
    # 15 Retalhuleu
    # 16 Sacatepéquez
    # 17 San marcos
    # 18 Santa rosa
    # 19 Sololá
    # 20 Suchitepéquez
    # 21 Totonicapán
    # 22 Zacapa
  dep=[
  [15.5, -90.333333],
  [15.1009234, -90.3139743],
  [14.6622, -90.8208],
  [14.8, -89.54583],
  [14.3579867773633, -89.84790854555509],
  [14.3009, -90.78581],
  [14.64072, -90.51327],
  [15.31918, -91.47241],
  [15.47225, -88.8407],
  [14.63472, -89.98889],
  [14.29167, -89.89583],
  [16.8, -89.93333],
  [14.83472, -91.51806],
  [15.03085, -91.14871],
  [14.53611, -91.67778],
  [14.578414124412356, -90.79401954555287],
  [14.96389 , -91.79444],
  [14.15015235, -90.3508818353375],
  [14.77222 , -91.18333],
  [14.37766785, -91.3643907717613],
  [14.91167 , -91.36111],
  [14.97222, -89.53056]]
  depto = []
  na = st.selectbox("Nombre del Estudiante: ", BecNames2,1)
  no=0
  for w in BecNames2:
    if w == na:
      break
    no+=1
  if na:
# tx= "Estudiante:  "+ na+ "\n pertenece al departamento de: \n "+BecLugar2[no] 
    # st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>', unsafe_allow_html=True)
    # st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">pertenece al departamento de: </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecLugar2[no]}</p>', unsafe_allow_html=True)


  for B in BecLugar2:
    if B == 'Alta Verapaz':
      depto.append(dep[0])
    elif B == 'Baja Verapaz':
      depto.append(dep[1])
    elif B == 'Chimaltenago':
      depto.append(dep[2])
    elif B == 'Chiquimula':
      depto.append(dep[3])
    elif B == 'El Progreso':
      depto.append(dep[4])
    elif B == 'Escuintla':
      depto.append(dep[5])
    elif B == 'Guatemala':
      depto.append(dep[6])
    elif B == 'Huehuetenango':
      depto.append(dep[7])
    elif B == 'Izabal':
      depto.append(dep[8])
    elif B == 'Jalapa':
      depto.append(dep[9])
    elif B == 'Jutiapa':
      depto.append(dep[10])
    elif B == 'Petén':
      depto.append(dep[11])
    elif B == 'Quetzaltenango':
      depto.append(dep[12])
    elif B == 'Quiché':
      depto.append(dep[13])
    elif B == 'Retalhuleu':
      depto.append(dep[14])
    elif B == 'Sacatepéquez':
      depto.append(dep[15])
    elif B == 'San Marcos':
      depto.append(dep[16])
    elif B == 'Santa Rosa':
      depto.append(dep[17])
    elif B == 'Sololá':
      depto.append(dep[18])
    elif B == 'Suchitepéquez':
      depto.append(dep[19])
    elif B == 'Totonicapán':
      depto.append(dep[20])
    elif B == 'Zacapa':
      depto.append(dep[21])

  data = pd.DataFrame(
    np.array(depto),
    # ['Alta Verapaz', 'Baja Verapaz', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango', 'Chimaltenango'],
    columns=['lat', 'lon'])
  st.map(data)

  
  # latlist = []
  # for d in depto:
  #   latlist.append(d[0])
  # lonlist = []
  # for d in depto:
  #   lonlist.append(d[1])
  # data = pd.DataFrame(
  #   BecNames2,
  #   latlist,
  #   lonlist)
  # st.write(data)
  # fig = px.scatter_mapbox(data, lat=latlist, lon=lonlist,size_max=15, zoom=10)
  # st.plotly_chart(fig)

def becedad2():
  global rowsreportes, rowsresformularios2
  na = st.selectbox("Nombre del Estudiante: ", BecNames2, 1)
  no = 0
  for w in BecNames2:
      if w == na:
          break
      no += 1
  if na:
      # st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>',
      #             unsafe_allow_html=True)
      # st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>',
      #             unsafe_allow_html=True)
      st.markdown(
          f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Tiene: </p>',
          unsafe_allow_html=True)
      st.markdown(
          f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecEdad2[no]} Años.</p>',
          unsafe_allow_html=True)

  st.subheader("Gráfico de Edades (Barras)")
  chart_data = pd.DataFrame({'Nombres': BecNames2, 'Edades': BecEdad2})
  st.bar_chart(chart_data.set_index('Nombres'))

  # st.subheader("VISTA 2 EDAD DE BECADOS (Sunburst)")
  # acumconts = [str(int(carne2[NamesAll2.index(w)])) for w in BecNames2]
  # acumcontstemp = acumconts
  # acumconts.pop()
  # data = dict(
  #     names=acumcontstemp,
  #     parent=acumconts.insert(0, ""),
  #     value=BecEdad2)

  # fig = px.sunburst(
  #     data,
  #     names='names',
  #     parents='parent',
  #     values='value',
  # )
  # st.plotly_chart(fig)

def becuni2():
  global rowsreportes2, rowsresformularios2
  hist_data = [x1, x2, x3]
  group_labels = ['Group 1', 'Group 2', 'Group 3']
  fig = ff.create_distplot(hist_data, group_labels, bin_size=[.1, .25, .5])
  st.plotly_chart(fig, use_container_width=True)

def becpro2():
  global rowsreportes2, rowsresformularios2

  # Obtener el nombre seleccionado
  na = st.selectbox("Nombre del Estudiante: ", BecNames2, 1)
  no = 0

  # Encontrar el índice del estudiante seleccionado
  for w in BecNames2:
      if w == na:
          break
      no += 1

  # Mostrar información del estudiante seleccionado
  if na:
      st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Esta trabajando en los ejes de: </p>', unsafe_allow_html=True)
      st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecTipo2[no]} </p>', unsafe_allow_html=True)

  # Crear una lista de listas para los datos de la tabla
  ndata = [['BECADOS', 'AREA DE TRABAJO', 'DEPTO DONDE TRABAJA']]

  # Llenar la lista con datos
  for b in BecNames2:
      ndata.append([b, BecTipo2[BecNames2.index(b)], Becdeptoreplic2[BecNames2.index(b)]])

  # Crear un DataFrame para la tabla
  tabla_datos = pd.DataFrame(ndata[1:], columns=ndata[0])

  # Mostrar la tabla en Streamlit
  st.subheader("Área de Trabajo y Departamento de Todos los Estudiantes")
  st.table(tabla_datos)


def bechoras2():
  global rowsreportes2, rowsresformularios2
  na = st.selectbox("Nombre del Estudiante: ", BecNames2,1)
  no=0
  for w in BecNames2:
    if w == na:
      break
    no+=1
  if na:
    # st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>', unsafe_allow_html=True)
    # st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">tiene Acumuladas: </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecHoras2[no]} Horas.</p>', unsafe_allow_html=True)

  st.subheader("HORAS POR BECADOS")
  # wide_df = px.data.medals_wide()
  # st.write(px.data.medals_wide())
  # st.write(type(px.data.medals_wide()))
  acumconts=[]
  for w in BecNames2:
    acumconts.append(str(int(carne2[NamesAll2.index(w)])))
  wide_df = {tuple(BecNames2),tuple(BecHoras2)}
  fig = px.bar(wide_df, x=acumconts, y=BecHoras2,color=BecNames2, title="HORAS REALIZADAS POR CADA BECADO (GRAFICO)")
  st.plotly_chart(fig)

  st.subheader("VISTA 2")
  acumcontstemp=acumconts
  acumconts.pop()
  data = dict(
    names=acumcontstemp,
    parent=acumconts.insert(0,""),
    value=BecHoras2)

  fig = px.sunburst(
      data,
      names='names',
      parents='parent',
      values='value',
  )
  st.plotly_chart(fig)

def becbeneficiadas2():
  global rowsreportes2, rowsresformularios2
  # Crear un diccionario para almacenar el total de personas beneficiadas por departamento
  total_personas_por_depto = {}

  for i in range(len(BecNames2)):
      becado_departamento = BecLugar2[i]
      personas_capacitadas = BecCapacitadas2[i]

      # Verificar si hay un valor numérico para personas capacitadas
      if isinstance(personas_capacitadas, (int, float)):
          # Sumar el número de personas capacitadas al total del departamento
          if becado_departamento in total_personas_por_depto:
              total_personas_por_depto[becado_departamento] += personas_capacitadas
          else:
              total_personas_por_depto[becado_departamento] = personas_capacitadas

  # Mostrar el total de personas beneficiadas por departamento
  st.subheader("Total de Personas Beneficiadas por Departamento")
  # for depto, total_personas in total_personas_por_depto.items():
  #     st.write(f"{depto}: {int(total_personas)} personas beneficiadas")

  # Crear un gráfico de barras para visualizar los totales por departamento

  # Convertir el diccionario a un DataFrame
  chart_data = pd.DataFrame(list(total_personas_por_depto.items()), columns=['Departamento', 'Total Personas'])

  # Crear una gráfica de barras con Plotly Express
  fig = px.bar(chart_data, x='Departamento', y='Total Personas', title='Total de Personas Beneficiadas por Departamento')

  # Mostrar la gráfica en Streamlit
  st.plotly_chart(fig)

    # Mostrar el total de personas beneficiadas por departamento en una tabla
  st.subheader("Total de Personas Beneficiadas por Departamento")

  # Crear un DataFrame con los datos
  tabla_datos = pd.DataFrame(list(total_personas_por_depto.items()), columns=['Departamento', 'Total Personas Beneficiadas'])

  # Formatear los números en el DataFrame
  tabla_datos['Total Personas Beneficiadas'] = tabla_datos['Total Personas Beneficiadas'].map('{:.0f}'.format)

  # Mostrar la tabla en Streamlit
  st.table(tabla_datos)
#! █████████████████████ OPERACIONES █████████████████████

def main():
  st.image('https://becas.usac.edu.gt/wp-content/uploads/2019/05/cropped-bannerN.png')
  st.title('Reporte de Servicio Social Becados SSE')
  st.write()
  st.sidebar.image('https://sgccc.org.gt/wp-content/uploads/2021/03/LOGO-USAC.-Rec-1.png')
  # Add a dropdown to select the year
  selected_year = st.sidebar.selectbox("Seleccione el Año:", ["2022", "2023", "2024"])


  st.markdown(
    """  <style>  
    span[data-baseweb="tag"] 
    {
    background-color: blue !important;  
    }  
    </style>  """,
  unsafe_allow_html=True,
  )
  # st.header('Bienvenidos :sunglasses:', anchor=None)

  # cfile()
  page_names_to_funcs = {}
  if selected_year=="2022":
    st.title('Becados 2022')
    page_names_to_funcs = {
      "Personas Beneficiadas": becbeneficiadas,
      "Becados por Departamento": becdep,
      "Becados por Edades": becedad,
      # "Becados por Unidad Academica": becuni,
      "Becados por Proyecto en el que trabaja": becpro,
      "Total de Horas por cada Becado": bechoras
    }
  elif selected_year=="2023":
    st.title('Becados 2023')
    page_names_to_funcs = {
      "Personas Beneficiadas": becbeneficiadas2,
      "Becados por Departamento": becdep2,
      "Becados por Edades": becedad2,
      # "Becados por Unidad Academica": becuni,
      "Becados por Proyecto en el que trabaja": becpro2,
      "Total de Horas por cada Becado": bechoras2
    }

  
  demo_name = st.sidebar.selectbox("Reportes: ", page_names_to_funcs.keys())
  page_names_to_funcs[demo_name]()
  #? Área para seleccionar las operaciones que desea realizar según lo seleccionado anteriormente.
  #? Área donde se puedan parametrizar los distintos algoritmos .
  #? Área donde se puedan visualizar de manera intuitiva los resultados. (como las gráficas).
if __name__ == "__main__":
    main()

