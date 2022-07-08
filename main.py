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

def pol():
  global colum, yaxe, df
  st.subheader('Regresión polinomial')
  #! REGRESION POLINOMIAL AREA DE RESULTADOS
  st.write("""La Regresión Polinomial es un caso especial de la Regresión Lineal, extiende el modelo lineal al agregar predictores adicionales, obtenidos al elevar cada uno de los predictores originales a una potencia. """)
  st.multiselect("Operaciones", [" Graficar puntos", "Definir función de tendencia", "Realizar predicción de la tendencia"])

  X= np.asarray(df[yaxe]).reshape(-1,1)
  y = np.asarray(df[colum]).reshape(-1,1)
  
  #Step 2: data preparation
  nb_degree =2
  fig, ax = plt.subplots(figsize=(5, 3))
  ax.scatter (X, y)
  polynomial_features = PolynomialFeatures(degree=nb_degree)
  X_TRANSF = polynomial_features.fit_transform (X)
  # Step 3: define and train a model
  model = linear_model.LinearRegression ()
  model.fit (X_TRANSF, y)
  #Step 4: calculate bias and variance
  Y_NEW =model.predict (X_TRANSF)
  rmse =np.sqrt (mean_squared_error(y, Y_NEW))
  r2 = r2_score (y, Y_NEW)
  print ('RMSE: ', rmse)
  print ('R2: ', r2)
  # plt.show()
  # st.pyplot(fig)
  
  nb_degree = st.number_input('Ingrese el grado de la funcion: .')
  nb_degree = st.slider('Ingrese el grado de la funcion:', 0, 15)
  
  # preTime = st.slider('Parametrizacion segun tiempo ingresado.',     int(X.min()), int(X.max()), (int(X.min()), int(X.max()) ))
  
  # preTime1 = int(preTime[0])
  # preTime2 = int(preTime[1])
  preTime1 = st.number_input('Parametrizacion segun tiempo 1 ingresados.')
  preTime2 = st.number_input('Parametrizacion segun tiempo 2 ingresados.')
  preTime1= int(preTime1)
  preTime2= int(preTime2)
  # if st.button('Prediccion Regresión polinomial'):
  # Step 5: prediction
  # Generar la figura.1
  X_NEW= np.linspace(preTime1, preTime2, preTime2)
  X_NEW = X_NEW[:, np.newaxis]
  X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)
  # st.write(f'Result: {model.predict(X_NEW_TRANSF)}')
  Y_NEW = model.predict(X_NEW_TRANSF)
  plt.plot (X_NEW, Y_NEW, color='coral', linewidth=3)
  plt.grid()
  plt.xlim (preTime1, preTime2)
  st.write("PREDICCION PARA: ",preTime2)
  st.write(Y_NEW[-1])
  adjust = st.slider('Ajustar Y.',     int(y.min())-1000, int(y.max())+1000, (int(y.min())-1000, int(y.max())+1000 ))
  adjustx = st.slider('Ajustar X.',     int(X.min())-1000, int(X.max())+1000, (int(X.min())-1000, int(X.max())+1000 ))
  preTime1=int(adjustx[0])
  preTime2=int(adjustx[1])
  plt.ylim (int(adjust[0]),int(adjust[1]))
  # st.write(model.predict([[2050]]))
  title = 'Degree = {}; RMSE = {}; R2 = {}'.format (nb_degree, round (rmse, 2), round(r2,2))
  plt.title("Polynomial Linear Regression using scikit-learn and python 3 \n "+title,fontsize=10)
  plt.xlabel( yaxe )
  plt.ylabel(colum)
  st.pyplot(fig)
  
  # Generar la figura.2
  regrprednew=[]
  regrprednew2=[]
  for i in Y_NEW.tolist():
    regrprednew.append(i[0])
  for J in X_NEW.tolist():
    regrprednew2.append(J[0])
  fig = px.scatter(y=df[colum],x=df[yaxe],  opacity=1)
  fig.add_traces(go.Scatter(x =regrprednew2, y =  regrprednew, name='Regression Fit'))
  st.plotly_chart(fig, use_container_width=True)
  st.subheader("FUNCION DE TENDENCIA: ")
  temp=""
  templist = model.coef_.tolist()

  templist = templist[0][::-1]
  le = len(templist)
  for element in range(len(templist)):
    temp+=" + "+str(templist[element])+"X^"+str(le)+" "
    le-=1
  funcion = "Y = " + temp
  st.write(funcion)
  st.success('Model trained successfully')

def tree():
  global colum, yaxe, df,header
  st.subheader('Clasificador de árboles de decisión')
  #! CLASIFICADOR GAUSSIANO AREA DE RESULTADOS
  st.write("""Un árbol de decisión o un árbol de clasificación es un árbol en el que cada nodo interno (no hoja) está etiquetado con una función de entrada. Los arcos procedentes de un nodo etiquetado con una característica están etiquetados con cada uno de los posibles valores de la característica.""")
  newlist=[]
  #! Creating labelEncoder
  le = preprocessing.LabelEncoder()
  for data in df:
    Tupla=tuple(le.fit_transform(df[data]))
    newlist.append(Tupla)
    print(newlist)
  newl=[]
  print(newlist[0])
  print(len(list(newlist[0]))-1)
  print(len(header)-1)
  for d in range(len(list(newlist[0]))):
    temp=[]
    for w in range(len(header)-1):
      if w ==0 :
        continue
      t=list(newlist[w])
      temp.append(t[d])
    st.write(str(temp))
    # print(temp)
    newl.append(tuple(temp))
  fig, ax = plt.subplots(figsize=(6, 4))
  clasify = st.selectbox("Nombre el header de clasificacion en el archivo.", header,1)
  title = st.text_input('Prediccion Gauss (Cuidado con la cantidad de variables)', 'Ingrese los valores correctos separados por coma (texto o numeros)')
  pred = title.split(",")
  predn = []
  try:
    for s in range(len(pred)):
      print(type(pred[s]))
      if isinstance(pred[s], str):
          predn.append(int(pred[s]))
  except:
    predn=le.fit_transform(pred)
  if st.button('Generar Arbol de Decision'):
    if title!='Ingrese los valores correctos separados por coma (texto o numeros)': 
      #! Combinig attributes into single listof tuples
      newlist.pop()
      newlist.pop(0)
      #! fit the model
      label = np.asarray(df[clasify])#! Elegir la columna E
      # label = np.where(label =='TRUE', 1, label)
      # label = np.where(label =='FALSE', 0, label)
      label = np.where(label =='N', 0, label)
      label = np.where(label =='P', 1, label)
      #Predict Output
      st.subheader("PREDICCION: ")
      clf = DecisionTreeClassifier().fit(newl, list(label))
      prediccion=clf.predict([predn])
      st.write(clf.predict([predn]))
      if prediccion == 0:
        st.write("N")
      else:
        st.write('P')
      # st.write(clf.predict(newl[predn]))
      plot_tree (clf, filled=True)
      st.pyplot(fig)
    else:
      st.warning("por favor ingrese valores de prediccion gracias :D")
    st.subheader("SIN PREDICCION: ")
  try:
    pass
    # #! Combinig attributes into single listof tuples
    # newlist.pop()
    # #! fit the model
    # label = np.asarray(df[clasify])#! Elegir la columna E
    # label = np.where(label =='TRUE', 1, label)
    # label = np.where(label =='FALSE', 0, label)
    # label = np.where(label =='N', 0, label)
    # label = np.where(label =='P', 1, label)
    # clf = DecisionTreeClassifier().fit(newl, list(label))
    # plot_tree (clf, filled=True)
    # st.pyplot(fig)
  except:
    pass

def gauss():
  global colum, yaxe, df,header
  st.header('Clasificador Gaussiano')
  #! CLASIFICADOR GAUSSIANO AREA DE RESULTADOS
  st.write("""El Clasificador de Procesos Gaussianos es un algoritmo de aprendizaje de la máquina de clasificación.
  Los procesos gausianos son una generalización de la distribución de probabilidad gausiana y pueden utilizarse como base de sofisticados algoritmos no paramétricos de aprendizaje automático para la clasificación y la regresión. """)
  newlist=[]
  #! Creating labelEncoder
  le = preprocessing.LabelEncoder()
  for data in df:
    Tupla=tuple(le.fit_transform(df[data]))
    newlist.append(Tupla)
  newl=[]
  print(newlist[0])
  print(len(list(newlist[0]))-1)
  print(len(header)-1)
  for d in range(len(list(newlist[0]))):
    temp=[]
    for w in range(len(header)-1):
      if w ==0 :
        continue
      t=list(newlist[w])
      temp.append(t[d])
    st.write(str(temp))
    newl.append(tuple(temp))
  fig, ax = plt.subplots(figsize=(4, 5))
  clasify = st.selectbox("Nombre el header de clasificacion en el archivo.", header,1)
  st.write(header)
  title = st.text_input('Prediccion Gauss (Cuidado con la cantidad de variables)', 'Ingrese los valores correctos separados por coma (texto o numeros)')
  pred = title.split(",")
  predn = []
  try:
    for s in range(len(pred)):
      print(type(pred[s]))
      if isinstance(pred[s], str):
          predn.append(int(pred[s]))
  except:
    predn=le.fit_transform(pred)

  if st.button('Generar Clasificacion Gaussiana'):
    # Create a Gaussian Classifier
    model=GaussianNB()

    #! Combinig attributes into single listof tuples
    newlist.pop()
    newlist.pop(0)
    #! fit the model
    st.write(clasify)
    label = np.asarray(df[clasify])#!

    # Train the model using the training sets
    model.fit(newl, label)
    # label = np.where(label =='TRUE', 1, label)
    # label = np.where(label =='FALSE', 0, label)
    label = np.where(label =='N', 0, label)
    label = np.where(label =='P', 1, label)
    #Predict Output
    # predicted= model.predict([predn]) # sunny, hot, high, false
    # print ("Predicted Value:", predicted)
    
    # if(predicted)
    #! fit the model
    # st.write(label)
    clf = DecisionTreeClassifier().fit(newl, list(label))
    plot_tree (clf, filled=True)
    st.write("PREDICCION: ")
    prediccion=clf.predict([predn])
    st.write(clf.predict([predn]))
    if prediccion == 0:
      st.write("N")
    else:
      st.write('P')
    st.pyplot(fig)
# MI FUNCION DE ACTIVACION
def sigmoide(x):
    return 1/(1 + np.exp(-x))

def redes():
  # Modelos
  # ==============================================================================
  # modelo_1 = MLPClassifier(
  #                 hidden_layer_sizes=(5),
  #                 learning_rate_init=0.01,
  #                 solver = 'lbfgs',
  #                 max_iter = 1000,
  #                 random_state = 123
  #             )

  # modelo_2 = MLPClassifier(
  #                 hidden_layer_sizes=(10),
  #                 learning_rate_init=0.01,
  #                 solver = 'lbfgs',
  #                 max_iter = 1000,
  #                 random_state = 123
  #             )

  # modelo_3 = MLPClassifier(
  #                 hidden_layer_sizes=(20, 20),
  #                 learning_rate_init=0.01,
  #                 solver = 'lbfgs',
  #                 max_iter = 5000,
  #                 random_state = 123
  #             )

  # modelo_4 = MLPClassifier(
  #                 hidden_layer_sizes=(50, 50, 50),
  #                 learning_rate_init=0.01,
  #                 solver = 'lbfgs',
  #                 max_iter = 5000,
  #                 random_state = 123
  #             )

  # modelo_1.fit(X=X, y=y)
  # modelo_2.fit(X=X, y=y)
  # modelo_3.fit(X=X, y=y)
  # modelo_4.fit(X=X, y=y)
  global colum, yaxe, df,header
  st.subheader('Redes neuronales')
  #! REDES NEURONALES AREA DE RESULTADOS
  st.write("""Una red neuronal es un método de la inteligencia artificial que enseña a las computadoras a procesar datos de una manera que está inspirada en la forma en que lo hace el cerebro humano.""")
  #? HIPERACTIVIDAD
  # X= np.asarray(df[yaxe]).reshape(-1,1)
  # y = np.asarray(df[colum]).reshape(-1,1)
  newlist=[]
  #! Creating labelEncoder
  le = preprocessing.LabelEncoder()
  for data in df:
    Tupla=tuple(le.fit_transform(df[data]))
    newlist.append(Tupla)
    print(newlist)
  newl=[]
  print(newlist[0])
  print(len(list(newlist[0]))-1)
  print(len(header)-1)
  for d in range(len(list(newlist[0]))):
    temp=[]
    for w in range(len(header)-1):
      if w ==0 :
        continue
      t=list(newlist[w])
      temp.append(t[d])
    st.write(str(temp))
    # print(temp)
    newl.append(tuple(temp))
  clasify = st.selectbox("Nombre el header de clasificacion en el archivo.", header,1)
  # title = st.text_input('Prediccion Gauss (Cuidado con la cantidad de variables)', 'Ingrese los valores correctos separados por coma (texto o numeros)')
  # pred = title.split(",")
  # predn = []
  # try:
  #   for s in range(len(pred)):
  #     print(type(pred[s]))
  #     if isinstance(pred[s], str):
  #         predn.append(int(pred[s]))
  # except:
  #   predn=le.fit_transform(pred)

  label = np.asarray(df[clasify])#! Elegir la columna E
  # label = np.where(label =='TRUE', 1, label)
  # label = np.where(label =='FALSE', 0, label)
  label = np.where(label =='N', 0, label)
  label = np.where(label =='P', 1, label)



  X_train, X_test, y_train, y_test = train_test_split(newl,list(label), random_state=1)
  scaler = preprocessing.StandardScaler().fit(X_train)
  X_train = scaler.transform(X_train)
  X_test = scaler.transform(X_test)
  # creating an classifier from the model:
  st.header("PARAMETRIZANDO: ")
  st.subheader("No. de neuronas: ")
  st.write("número de neuronas en cada capa oculta: (5, ) haría referencia a una capa oculta de 5 neuronas (como hemos comentado, la coma es necesaria para no confundir el número con un simple entero), y (50, 20, 10) haría referencia a tres capas ocultas de 50, 20 y 10 neuronas artificiales respectivamente.")
  capa1 = st.slider('Capa 1 .', 0, 250, 10)
  capa2 = st.slider('Capa 2 .', 0, 250, 10)
  capa3 = st.slider('Capa 3 .', 0, 250, 10)
  st.write("[",capa1,",",capa2,",",capa3,"]")
  clf = MLPClassifier(hidden_layer_sizes=(capa1, capa2, capa3), max_iter=16000)

  clf.fit(X_train, y_train)
  #! Creating labelEncoder
  le = preprocessing.LabelEncoder()
  # let's fit the training data to our model
  from sklearn.metrics import accuracy_score
  model1_train = clf.predict(X_train)
  model1_test = clf.predict(X_test)
  st.header("PREDICCIONES")
  title3 = st.text_input('Prediccion RED NEURONAL (Cuidado con la cantidad de variables)', 'Ingrese los valores correctos separados por coma (texto o numeros)  ')
  pred = title3.split(",")
  predn = []
  try:
    for s in range(len(pred)):
      if isinstance(pred[s], str):
          predn.append(int(pred[s]))
  except:
    predn=le.fit_transform(pred)
  # predicted= clf.predict([predn]) # sunny, hot, high, false
  # print ("Predicted Value:", predicted)
  if st.button('Generar PREDICCION: '):
    prediccion=clf.predict([predn])
    st.write(clf.predict([predn]))
    if prediccion == 0:
      st.write("N")
    else:
      st.write('P')
    st.subheader("PUNTUACION DE PRECISION: ")
    st.write(str(accuracy_score(model1_train, y_train)))
    # model1_test = clf.predict(X_test)
    st.write(str(accuracy_score(model1_test, y_test)))

    # fig, ax = plt.subplots(figsize=(4, 5))
    # plt.scatter(X_train, y_train,c = plt.rcParams['axes.prop_cycle'].by_key()['color'][1],marker    = 'o',
    #           edgecolor = 'black')
    # plot_tree (clf, filled=True)
    # st.pyplot(fig)
    # fig, ax = plt.subplots(figsize=(4, 5))
    # plot_tree (clf, filled=True)
    # st.pyplot(fig)

def alg():
  #*   Los algoritmos que el estudiante debe implementar en la aplicación son:
  #*            - Regresión lineal.
  #*            - Regresión polinomial.
  #*            - Clasificador Gaussiano.
  #*            - Clasificador de árboles de decisión.
  #*            - Redes neuronales.
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
  st.header('Bienvenida a mi , *Proyecto!* :sunglasses:', anchor=None)

  #? Carga de archivos. (Puede ser csv, xls, xlsx o json únicamente).
  cfile()
  #*   La aplicación debe poder realizar las siguientes operaciones:
  #*               - Graficar puntos.
  #*               - Definir función de tendencia (lineal o polinomial).
  #*               - Realizar predicción de la tendencia (según unidad de tiempo ingresada).
  #*               - Clasificar por Gauss o árboles de decisión o redes neuronales.
  #? Área para seleccionar el algoritmo que se desee ejecutar según archivo de entrada.
  alg()
  page_names_to_funcs = {
    "Regresion Lineal": regr,
    "Regresión polinomial": pol,
    "Clasificador de árboles de decisión": tree,
    "Clasificador Gaussiano": gauss,
    "Redes neuronales": redes
  }

  demo_name = st.sidebar.selectbox("Algoritmos: ", page_names_to_funcs.keys())
  page_names_to_funcs[demo_name]()
  #? Área para seleccionar las operaciones que desea realizar según lo seleccionado anteriormente.
  #? Área donde se puedan parametrizar los distintos algoritmos .
  #? Área donde se puedan visualizar de manera intuitiva los resultados. (como las gráficas).
if __name__ == "__main__":
    main()

