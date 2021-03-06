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
  uploaded_file = st.file_uploader("Puede ser csv, xls, xlsx o json ├║nicamente")
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

#! ÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľł ALGORITMOS ÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľł
def regr():
  global colum, yaxe, df
  st.subheader('Graficar Regresi├│n lineal')
  #! REGRESION LINEAL AREA DE RESULTADOS
  st.write("""La regresi├│n lineal es una t├ęcnica de modelado estad├şstico que se emplea para describir una variable de respuesta continua como una funci├│n de una o varias variables predictoras. Puede ayudar a comprender y predecir el comportamiento de sistemas complejos o a analizar datos experimentales, financieros y biol├│gicos.""")
  st.multiselect("Operaciones", [" Graficar puntos", "Definir funci├│n de tendencia", "Realizar predicci├│n de la tendencia"])

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
  if st.button('Regresi├│n lineal'):
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
    BecEdad.append(int(row[8]))

Becdeptoreplic = [] #* DEPTO DE REPLICA
for row in rowsresformularios:
    Becdeptoreplic.append(row[9])

BecCapacitadas = [] #* PERSONAS CAPACITADAS
for row in rowsresformularios:
    BecCapacitadas.append(row[10])

def becdep():
  global rowsreportes, rowsresformularios
  # st.write(rowsresformularios)
  # *La divisi├│n pol├ştica de Guatemala consta de 22 Departamentos:
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
    # 12 Pet├ęn
    # 13 Quetzaltenango
    # 14 Quich├ę
    # 15 Retalhuleu
    # 16 Sacatep├ęquez
    # 17 San marcos
    # 18 Santa rosa
    # 19 Solol├í
    # 20 Suchitep├ęquez
    # 21 Totonicap├ín
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
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>', unsafe_allow_html=True)
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
    elif B == 'Pet├ęn':
      depto.append(dep[11])
    elif B == 'Quetzaltenango':
      depto.append(dep[12])
    elif B == 'Quich├ę':
      depto.append(dep[13])
    elif B == 'Retalhuleu':
      depto.append(dep[14])
    elif B == 'Sacatep├ęquez':
      depto.append(dep[15])
    elif B == 'San Marcos':
      depto.append(dep[16])
    elif B == 'Santa Rosa':
      depto.append(dep[17])
    elif B == 'Solol├í':
      depto.append(dep[18])
    elif B == 'Suchitep├ęquez':
      depto.append(dep[19])
    elif B == 'Totonicap├ín':
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
  na = st.selectbox("Nombre del Estudiante: ", BecNames,1)
  no=0
  for w in BecNames:
    if w == na:
      break
    no+=1
  if na:
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">tiene: </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecEdad[no]} A├▒os.</p>', unsafe_allow_html=True)

  st.subheader("Grafico de Edades")
  # chart_data = pd.DataFrame(
  # np.array(BecEdad),
  # columns=BecNames)
  # st.bar_chart(chart_data)
  # Add histogram data
  
  newedad=[]
  for ns in BecEdad:
    newedad.append(ns)
  df = pd.DataFrame(  np.array([newedad]), columns =BecNames)
  # Plot!
  st.bar_chart(df)
  st.subheader("VISTA 2 EDAD DE BECADOS")
  acumconts=[]
  for w in BecNames:
    acumconts.append(str(int(carne[NamesAll.index(w)])))
  acumcontstemp=acumconts
  acumconts.pop()
  data = dict(
    names=acumcontstemp,
    parent=acumconts.insert(0,""),
    value=newedad)

  fig = px.sunburst(
      data,
      names='names',
      parents='parent',
      values='value',
  )
  st.plotly_chart(fig)

def becuni():
  global rowsreportes, rowsresformularios

  hist_data = [x1, x2, x3]

  group_labels = ['Group 1', 'Group 2', 'Group 3']
  fig = ff.create_distplot(
         hist_data, group_labels, bin_size=[.1, .25, .5])
  st.plotly_chart(fig, use_container_width=True)
def becpro():
  global rowsreportes, rowsresformularios
  na = st.selectbox("Nombre del Estudiante: ", BecNames,1)
  no=0
  for w in BecNames:
    if w == na:
      break
    no+=1
  if na:
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Esta trabajando en los ejes de: </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{BecTipo[no]} </p>', unsafe_allow_html=True)
  ndata=[['BECADOS', 'AREA DE TRABAJO', 'DEPTO DONDE TRABAJA']]
  for b in BecNames:
    ndata.append([b,BecTipo[BecNames.index(b)],Becdeptoreplic[BecNames.index(b)]])
  st.write(ndata)
  fig =  ff.create_table(ndata)
  st.plotly_chart(fig)
def bechoras():
  global rowsreportes, rowsresformularios
  na = st.selectbox("Nombre del Estudiante: ", BecNames,1)
  no=0
  for w in BecNames:
    if w == na:
      break
    no+=1
  if na:
    st.markdown(f'<p style="background-color:#0;color:#05000A;font-size:24px;border-radius:2%;">Estudiante:  </p>', unsafe_allow_html=True)
    st.markdown(f'<p style="background-color:#F0FF00;color:#05000A;font-size:24px;border-radius:2%;">{na}</p>', unsafe_allow_html=True)
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

#! ÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľł OPERACIONES ÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľłÔľł

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
  page_names_to_funcs = {
    "Becados por Departamento": becdep,
    "Becados por Edades": becedad,
    # "Becados por Unidad Academica": becuni,
    "Becados por Proyecto en el que trabaja": becpro,
    "Total de Horas por cada Becado": bechoras
  }

  st.sidebar.image('https://www.redfia.net.gt/wp-content/uploads/2019/09/LOGO-USAC-2012-1.png')
  demo_name = st.sidebar.selectbox("Algoritmos: ", page_names_to_funcs.keys())
  page_names_to_funcs[demo_name]()
  #? ├ürea para seleccionar las operaciones que desea realizar seg├║n lo seleccionado anteriormente.
  #? ├ürea donde se puedan parametrizar los distintos algoritmos .
  #? ├ürea donde se puedan visualizar de manera intuitiva los resultados. (como las gr├íficas).
if __name__ == "__main__":
    main()

