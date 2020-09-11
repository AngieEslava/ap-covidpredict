import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime , timedelta
from load_api import(load_api)
import matplotlib.pyplot as plt

@st.cache
def load_data():
    raw = pd.read_csv('data.csv')
    #raw = load_api()
    return raw
data = load_data()
Dataset_fallecidos = data[data['Atención'] == 'Fallecido']

###---------Titulos principales -------------------
st.write("""
# Proyecto Covid-19 Analítica Predictiva
*Saul Hernandez, Angie Eslava, Steven Salgado*
""")
st.sidebar.header('Navegacion')

nav_link = st.sidebar.radio(" ", ("Inicio", "Análisis Exploratorio de Infectados", "Análisis Exploratorio de Fallecidos", "Modelo Predictivo", "Conclusiones"))

if nav_link == "Inicio":
    st.write('En el mundo actual, la pandemia causada por el Covid-19 es y será por mucho tiempo el tema mas comentado e importante. Colombia no ha sido extraño a este fenómeno que afecta hoy la vida de millones de colombianos a través de toda la región. En este trabajo nos enfrentamos a la tarea de abordar la enfermedad desde los datos.')
    st.write('La información utilizada es tomada de  [click aquí] (https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia/gt2j-8ykr/data)')
    st.write('A continuación se muestra un dashboard con el analisis de datos por ciudad realizado en Google Data Studio, (Actualizado a la fecha de domingo 06/09/2020)')
    st.header(
        "Dashboard Google Data Studio"
    )
    st.markdown("""
    <iframe width="600" height="450" src="https://datastudio.google.com/embed/reporting/88ee007b-8f97-4cee-87ae-e47dd20fcb41/page/PzjeB" frameborder="0" style="border:0" allowfullscreen></iframe> 
   """, unsafe_allow_html=True)
elif nav_link == "Análisis Exploratorio de Infectados":
    st.header("Análisis Exploratorio de Datos")
    st.write('A continuación se muestran una serie de gráficos que despliegan información acerca de los datos de inspección de Covid-19 de las cinco ciudades principales de Colombia')
    # Estado actual de pacientes por sexo
    Data_Female = data[data['Sexo'] == 'F']
    Data_Male = data[data['Sexo'] == 'M']
    #Ancho de las barras
    bar_width = 0.30
    #Valores de las barras
    Barras_hombre = Data_Male['Atención'].value_counts()
    Barras_mujer = Data_Female['Atención'].value_counts()
    #Valores en el eje x
    r1 = np.arange(len(Barras_hombre))
    r2 = [x + bar_width for x in r1]

    st.write('**Estado de los infectados por sexo**')
    #Graficas
    plt.bar(r1, Barras_hombre, color='darkturquoise', width=bar_width, edgecolor='white', label='Hombres')
    plt.bar(r2, Barras_mujer, color='powderblue', width=bar_width, edgecolor='white', label='Mujeres')
    #Nombre eje x
    plt.xlabel('Estado del paciente ', fontweight='bold')
    #Reemplazo de los valores del eje x por los nombres de la columna Atención
    plt.xticks([r + bar_width-0.15 for r in range(len(Barras_hombre))], Barras_hombre.index, rotation = 'vertical')
    # Definición de tamaño de gráfica
    fig = plt.gcf()
    fig.set_size_inches(8, 5)
    plt.gca().spines["left"].set_color("lightgray")
    plt.gca().spines["bottom"].set_color("gray")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    #Visualización
    plt.legend()
    plt.show()
    st.pyplot()
    st.write('Se puede observar que la mayor cantidad de pacientes se encuentran recuperados, lo que es una buena señal sobre la letalidad del virus, ya que la mayor cantidad de los pacientes sobrevive el virus. Se observa también, que hay ligeramente más mujeres recuperadas, pero más hombres tanto fallecidos como en hospital')

    st.write('**Distribución Infectados**')
    # Distribucion por sexo de pacientes
    tamaño = data['Sexo'].value_counts()
    labels = data['Sexo'].value_counts().index
    # Distribucion por sexo de pacientes
    fig1, ax1 = plt.subplots()
    ax1.pie(tamaño, labels=labels, autopct='%1.f%%',
            shadow=False, startangle=90, colors = ['aqua' , 'coral'])
    ax1.axis('equal')  # Equal asegura que el diagrama sea circular.
    fig1.set_size_inches(6, 2)
    plt.show()
    st.pyplot()
    st.write('Luego realizamos una distribución de cantidad de infectados por sexo, se observa que la distribución en las cinco ciudades es completamente equitativa entre hombres y mujeres, aunque se sabe que por literatura los hombres son un poco mas susceptibles a ser infectados')
    
    
    # Ciudad de ubicación de pacientes
    st.write('**Distribución de Infectados por Ciudad**')
    Data = data['Ciudad de ubicación'].value_counts()
    r1 = np.arange(len(Data))
    plt.bar(r1, Data, color='lightsteelblue', edgecolor='white' , label ='Infectados por ciudad')
    plt.xticks([r for r in range(len(Data))] , Data.index, rotation = 'vertical')
    plt.legend()
    plt.gca().spines["left"].set_color("lightgray")
    plt.gca().spines["bottom"].set_color("gray")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.show()
    st.pyplot()
    st.write('Se revisa la distribución de infectados por ciudad, se encuentra una distribución acorde al tamaño poblacional de las ciudades, aunque Barranquilla esta bastante cerca de Cali, teniendo la primera la mitad de población que la segunda. ')

    # Visualización de categorias por edad
    st.write('**Visualización de categorias por edad**')
    Data = data['Cat_Edad'].value_counts()
    r1 = np.arange(len(Data))
    plt.barh(r1, Data, color='salmon', edgecolor='white' , label = 'Infectados por categoria de edad')
    plt.yticks([r for r in range(len(Data))] , Data.index, rotation = 'horizontal')
    plt.gca().spines["left"].set_color("gray")
    plt.gca().spines["bottom"].set_color("lightgray")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(8, 6)
    plt.show()
    st.pyplot()
    st.write('Se revisa la cantidad de infectados por rango de edad, se observa que la mayor cantidad de infectados se encuentra en la categoria de adulto, entre los rangos de edad de 26 a 59, esto puede asociarse a que a medida que incrementa la edad las personas son mas susceptibles al virus, ademas de que esta categoria es la que posee un mayor rango respecto a las demás')

    #Casos importados vs contraidos en territorio nacional
    st.write('**Casos importados vs contraidos en territorio nacional**')
    Data = data['País de procedencia'].value_counts()[:5]
    r1 = np.arange(len(Data))
    plt.barh(r1, Data, color='sandybrown', edgecolor='white')
    plt.yticks([r for r in range(len(Data))] , Data.index, rotation = 'horizontal')
    plt.gca().spines["left"].set_color("gray")
    plt.gca().spines["bottom"].set_color("lightgray")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend()
    plt.show()
    st.pyplot()
    st.write('Por último se visualiza para los casos importados, que países son los que más casos han aportado a nuestro país, se observa que España y Estados Unido son los paises que mas aportan por una gran diferencia respecto a los demás países. ')

elif nav_link == "Análisis Exploratorio de Fallecidos":
    st.write('Para solucionar la pregunta de analítica acerca de la predicción de casos  de fallecidos, se hace primero un analisis exploratorio de datos filtrando por aquellos pacientes que han fallecido')
    # Análisis basico de los fallecidos 
    st.write('**Distribución por sexo de pacientes**')
    tamaño = Dataset_fallecidos['Sexo'].value_counts()
    labels = Dataset_fallecidos['Sexo'].value_counts().index
    fig1, ax1 = plt.subplots()
    ax1.pie(tamaño, labels=labels, autopct='%1.f%%',
            shadow=False, startangle=90, colors = ['coral' , 'aqua'])
    ax1.axis('equal')  # Equal asegura que el diagrama sea circular.
    fig = plt.gcf()
    fig.set_size_inches(6,2)
    plt.show()
    st.pyplot()
    st.write('Se encuentra que la distribución de fallecidos respecto a sexo, difiere considerablemente de la distribución de infectados, ya que aunque el primero era totalmente equitativo, en este se ve una predominancia de los hombres, esto se asocia tanto  a razones genéticas expresadas en la literatura,como a razones sociales en menor medida, ya que los hombres pueden incurrir en mayores descuidos respecto al cuidado del virus')

    #Análisis de fallecidos por ciudad
    st.write('**Análisis de fallecidos por ciudad**')
    Data = Dataset_fallecidos['Ciudad de ubicación'].value_counts()
    r1 = np.arange(len(Data))
    plt.barh(r1, Data, color='palegreen', edgecolor='white' , label = "Fallecidos por Ciudad")
    plt.yticks([r for r in range(len(Data))] , Data.index, rotation = 'horizontal')
    plt.xlabel("Cantidad de fallecidos")
    plt.gca().spines["left"].set_color("gray")
    plt.gca().spines["bottom"].set_color("lightgray")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend()
    plt.show()
    st.pyplot()
    st.write('Se puede observar que la ciudad con mayor cantidad de fallecidos es nuestra capital Bogotá, lo que es lógico debido al tamaño y densidad poblacional de la ciudad con  7.4 M de habitantes. Sin embargo el segundo dato resulta curioso, ya que es Barranquilla la ciudad que le sigue en cantidad de fallecidos, la cual tiene menor cantidad de habitantes (1.2 M) que Cali (2.2 M) o Medellín (2.4M). Esto se puede deber a distribucion de la población, problemas en el sistema de salud, o a cuestiones culturales respecto al manejo personal de cuidados contra el Covid-19')
    
    #Distribución de Fallecidos por Categoría de edad
    st.write('***Distribución de Fallecidos por Categoría de Edad***')
    Data = Dataset_fallecidos['Cat_Edad'].value_counts()
    r1 = np.arange(len(Data))
    plt.barh(r1, Data, color='khaki', edgecolor='white' , label = "Fallecidos por rango de edad")
    plt.yticks([r for r in range(len(Data))] , Data.index, rotation = 'horizontal')
    plt.xlabel("Cantidad de fallecidos")
    plt.gca().spines["left"].set_color("gray")
    plt.gca().spines["bottom"].set_color("lightgray")
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.legend()
    st.pyplot()
    st.write('En la distribución por rango de edad se observa claramente que la teoría de que los adultos mayores son mas susceptibles a fallecer por la enfermedad es cierta, ya que la mayoria de fallecidos está entre aquellos mayores a 59 años, teniendo una gran cantidad también en la categoría de adultos que cubre entre los 29 y 59 años. ')

elif nav_link == "Modelo Predictivo":
    st.write(
    """## Modelo de Predicción Random Forest
    """)
    #%----------------Dataset y modelado para predicción----------------------
    # Se agrupa por Fecha de Muerte para conocer la cantidad de fallecidos por día
    Fallecidos_por_Fecha = Dataset_fallecidos.groupby("Fecha de muerte")['Sexo'].count().reset_index()
    # Se realiza un cambio de tipo de la columna fechas a str, ya que este tipo es necesario par aun proceso de merge posterior.
    Fallecidos_por_Fecha['Fecha de muerte'] = Fallecidos_por_Fecha['Fecha de muerte'].astype(str)
    Fallecidos_por_Fecha.columns = ['Fecha' , 'Fallecidos']

    #%------------------------------------------------------------------------
    #Para solucionar la falta de fechas, se realiza el siguiente procedimiento
    #Se crean fechas iniciales y finales
    #Para definir la fecha del día final, se toma el ultimo valor del dataset, y se separa en sus componentes
    Año = int((str(Fallecidos_por_Fecha['Fecha'][-1:]).split('-')[0]).split('    ')[1])
    Mes = int(str(Fallecidos_por_Fecha['Fecha'][-1:]).split('-')[1])
    Dia = int(str(Fallecidos_por_Fecha['Fecha'][-1:]).split('-')[2][0:2])

    # Se crean valores para la fecha inicial (que siempre es la misma) y la fecha final del dataset
    Fecha_inicial = datetime(2020,3,16 )
    Fecha_final =  datetime(Año,Mes,Dia)

    #Se crea un listado de fechas que contenga todos los días sin saltarse ninguno
    Lista_fechas = []
    fecha = Fecha_inicial
    days = 0 
    while fecha < Fecha_final:
        fecha = Fecha_inicial +  timedelta(days)
        days += 1 
        Lista_fechas.append(fecha)

    #Se crea un dataset que contenga el listado de fechas completas
    Df_fechas = pd.DataFrame(Lista_fechas)

    #Se redefine el tipo de las fechas para poder hacer un Merge y se renombra 
    Df_fechas[0] = Df_fechas[0].astype(str)
    Df_fechas.columns = ['Fecha']

    #Se crea un Dataset final de fechas que contiene el Merge del dataset creado y el de fechas de fallecidos
    Fechas_final = Df_fechas.merge(Fallecidos_por_Fecha , on= 'Fecha' , how = 'left')
    Fechas_final.fillna(0,inplace=True)

    #Se cambia el tipo de la columna fallecidos a Int
    Fechas_final['Fallecidos'] = Fechas_final['Fallecidos'].astype(int)

    #Se convierte la columna de fallecidos en fallecidos hoy, para poder identificarla despues del shift
    Fechas_final.columns = ['Fecha' , 'Fallecidos hoy']

    #Se crean las columnas de fallecidos de ayer y fallecidos de mañana respectivamente
    Fechas_final['Fallecidos ayer'] = Fechas_final['Fallecidos hoy'].shift(periods = 1)
    Fechas_final['Fallecidos mañana'] = Fechas_final['Fallecidos hoy'].shift(periods = -1)

    #Se visualiza a continuación el dataset con las columnas necesarias para la predicción
    Fechas_final['Diferencia_hoy_ayer'] = Fechas_final['Fallecidos hoy'].diff()
    #Fechas_final.iloc[:,0:4]

    #Se crea una función donde se ingresa el dataset base del que se va a tomar, 
    # y el número de días que han de transcurrir entre fecha y fecha
    @st.cache
    def Dataset_por_tiempo(Df_base, days):
        Df = pd.DataFrame()
        for i in range(1, Df_base.index.size, days):
            Df = Df.append(Df_base.loc[[i]])
            Df = Df.reset_index(drop = True)
            Df['Diferencia_hoy_ayer'] = Df['Fallecidos hoy'].diff()
        return Df

    ### ****** Predicción con Random Forest  **********
    #Se define una función de preprocesamiento, que trata el dataframe, y otorga el set de entrenamiento y testeo
    @st.cache
    def Preprocesamiento(df):
        #Reorganizo el dataset dejando como columna final la columna de predicción que serán los fallecidos de mañana
        df = df[['Fallecidos hoy','Fallecidos ayer', 'Diferencia_hoy_ayer','Fallecidos mañana']]
        # Se quitan la primera y ultima fila ya que no se puede tratar con los valores NaN
        df = df.iloc[1:-3]
        # Para definir el set de entrenamiento se toma el 90% inicial como train y el 10% final como test
        Var_90 = int(df.index.size*0.90)
        X = df.drop(['Fallecidos mañana'] , axis = 1)
        y = df['Fallecidos mañana']

        X_train = (X[X.index < Var_90]).astype(int)
        y_train = (y[y.index < Var_90]).astype(int)             
            
        X_test = (X[X.index >= Var_90]).astype(int)    
        y_test = (y[y.index >= Var_90]).astype(int)
            
        return X_train, y_train, X_test, y_test

    @st.cache
    def modeltrain(X_train, y_train, X_test, y_test):
        from sklearn.ensemble.forest import RandomForestRegressor
        # Generando el modelo 
        RF_Model = RandomForestRegressor(n_estimators=100,max_features=1)
        # Ajustando el modelo con X_train y y_train
        rgr = RF_Model.fit(X_train, y_train)
        y_train_predict = (rgr.predict(X_train)).astype(int)
        y_test_predict = (rgr.predict(X_test)).astype(int)
        
        return y_train_predict ,  y_test_predict , rgr

    @st.cache
    def prediction(df, X_testing , days, model):
        # Se define un dataset donde se adicionaran
        Data_prediccion = X_testing.iloc[[-1]]
        list_tomorrow = []
        index_list = []
        for i in range(days):
            index_list.append(df.index.size + i)
            tomorrow = model.predict(Data_prediccion.iloc[[-1]]).astype(int)[0]
            list_tomorrow.append(tomorrow)
            ayer = Data_prediccion.iloc[-1, 0]
            Diff = tomorrow-ayer
            Data_prediccion = Data_prediccion.append({"Diferencia_hoy_ayer" : Diff , "Fallecidos ayer" : ayer , "Fallecidos hoy" : tomorrow}, 
                                                    ignore_index=True)
        Prediccion_df = pd.DataFrame(index = index_list , data = list_tomorrow, columns = ["Fallecidos_predict"])

        y = df['Fallecidos mañana']
        
        return Prediccion_df , index_list, list_tomorrow, y

    # Procedimiento para dataset por Día
    X_train_dia, y_train_dia, X_test_dia, y_test_dia = Preprocesamiento(Fechas_final)

    #grafica
    st.write('**Predicción por día **')
    plt.figure(1)
    plt.plot(y_train_dia.index , y_train_dia, label = "Set de entrenamiento", color = 'seagreen')
    plt.plot(y_test_dia.index , y_test_dia, label = "Set de testeo", color = 'limegreen')    
    plt.xlabel('Días desde inicio de la pandemia')
    plt.ylabel('Cantidad de fallecidos')
    plt.legend()
    plt.show()
    st.pyplot()
    st.write('Se observa la división entre el set de entrenamiento que contiene el 90% inicial de los datos, y el set de testeo, que contiene el 10% restante,la división se realiza de manera continua y no aleatoria, por ser una predicción de serie de tiempo')

    # Entrenamiento y visualización del comportamiento en el set de testing
    y_train_predict_dia ,  y_test_predict_dia , rgr_dia = modeltrain(X_train_dia, y_train_dia, X_test_dia, y_test_dia)

    #Graficando el 90% real, y el 10% tanto de testing como de predicción, para encontrar el ajuste gráfico
    #Gráfica del 90% real
    st.write('**Predicción del set de testeo **')
    plt.figure(2)
    plt.plot(y_train_dia.index , y_train_dia, label = "90% Real", color = 'seagreen')
    plt.plot(y_test_dia.index, y_test_dia, label = "10% Testeo", color = 'limegreen')    
    plt.plot(y_test_dia.index, y_test_predict_dia, label = "10% Predicho",color = 'indianred')
    plt.xlabel('Días desde inicio de la pandemia')
    plt.ylabel('Cantidad de fallecidos')
    plt.legend()
    plt.show()
    st.pyplot()
    st.write('Se realiza un entrenamiento del modelo con el set de entrenamiento que contiene el 90% inicial. Y se realiza la predicción para el set de testing, se observa que en la figura en verde la predicción, aun cuando no tiene una precisión alta, mantiene claramente la tendencia de los datos reales')

    # Predicción para 15 días 
    st.write('**Predicción de 15 días **')
    Predicción_dia, indice_dia , tomorrow_dia, y_dia = prediction(Fechas_final, X_test_dia , 15 , rgr_dia)
    plt.figure(3)
    plt.plot(y_dia.index , y_dia, label = "Real", color = 'seagreen')
    plt.plot(indice_dia, tomorrow_dia ,  label = "Predicción", color = 'red')
    plt.xlabel('Días desde inicio de la pandemia')
    plt.ylabel('Cantidad de fallecidos')
    plt.legend()
    plt.show()
    st.pyplot()
    st.write('Por último se realiza la predicción para los siguientes 15 días, se observa que el modelo indica un nuevo incremento en los casos (aunque en diferentes corridas con las mismas variables puede cambiar su pendiente). Se cree que el modelo aun cuando no es preciso, puede estar prediciendo una tendencia de crecimiento en los casos de fallecidos. Con el entrenamiento pudo haber aprendido que de pocos casos se empieza a pasar a un incremental de casos como sucede alrededor de los 90 días en la data rereal. Esto puede verse acorde con la reapertura económica realizada en el mes de septiembre, que con una alta probabilidad aumentará el numero de casos, y con esto el número de fallecidos')

    # Procedimiento para dataset por semana
    Df_semanal = Dataset_por_tiempo(Fechas_final, 7)
    X_train_semana, y_train_semana, X_test_semana, y_test_semana = Preprocesamiento(Df_semanal)

    #grafica
    st.write('**Predicción Semanal **')
    st.write('*Set de Testeo y Set de Entrenamiento *')
    plt.figure(4)
    plt.plot(y_train_semana.index , y_train_semana, label = "Set de entrenamiento", color = 'seagreen')
    plt.plot(y_test_semana.index , y_test_semana, label = "Set de testeo", color = 'limegreen')    
    plt.xlabel('Días desde inicio de la pandemia')
    plt.ylabel('Cantidad de fallecidos')
    plt.legend()
    plt.show()
    st.pyplot()
    st.write('Se realiza la separación del set de testeo y el set de entrenamiento y se observa que debido a la división por semana la cantidad de datos se reduce de manera significativa')

    # Entrenamiento y visualización del comportamiento en el set de testing
    y_train_predict_semana ,  y_test_predict_semana , rgr_semana = modeltrain(X_train_semana, y_train_semana, X_test_semana, y_test_semana)

    #Graficando el 90% real, y el 10% tanto de testing como de predicción, para encontrar el ajuste gráfico
    #Gráfica del 90% real
    st.write('** Prediccion Set de Testeo por Semana **')
    plt.figure(5)
    plt.plot(y_train_semana.index , y_train_semana, label = "90% Real", color = 'seagreen')
    plt.plot(y_test_semana.index, y_test_semana, label = "10% Testeo", color = 'limegreen')    
    plt.plot(y_test_semana.index, y_test_predict_semana, label = "10% Predicho",color = 'indianred')
    plt.xlabel('Semanas desde inicio de la pandemia')
    plt.ylabel('Cantidad de fallecidos')
    plt.legend()
    plt.show()
    st.pyplot()
    st.write('Se observa la predicción del modelo en la data de testeo, es claro que no es para nada adecuada, esto es muy probablemente a la reducción en la cantidad de datos al recortar el dataset de días a semanas. ')

    # Predicción para 15 días 
    st.write('** Predicción Semanal **')
    plt.figure(6)
    Predicción_semana, indice_semana , tomorrow_semana, y_semana = prediction(Df_semanal, X_test_semana , 4 , rgr_semana)
    plt.plot(y_semana.index , y_semana, label = "Real", color = 'seagreen')
    plt.plot(indice_semana, tomorrow_semana, label = "Predicción", color = 'red')
    plt.xlabel('Días desde inicio de la pandemia')
    plt.ylabel('Cantidad de fallecidos')
    plt.legend()
    plt.show()
    st.pyplot()
    st.write('Se observa que la predicción en este caso resulta más pobre que la realizada por día con el dataset completo,  hay una especia de estabilid con una ligera tendencia de aqui a 4 semanas, es decir un mes')

elif nav_link == "Conclusiones":
    st.write("""
    ## Conclusiones
    *   En la sección de análisis exploratorio se encuentran  2 datos curiosos, el primero que la distribución de hombres y mujeres esta llegando a una estabilidad de 50/50 infectados, pero la distribución de fallecidos es de 65% para hombres y 35% para mujeres. Lo que indica una tasa de mortalidad mayor para los hombres, como se ha mencionado en noticias y literatura.
    *   Se encuentra que la distribución de infectados y fallecidos por ciudades corresponde en cierta medida al tamaño por ciudad, excepto en Barranquilla, donde a pesar de tener una población menor que Medellín y Cali, presenta mayor cantidad de infectados y fallecidos, esto asociado a problemas en el sistema de salud, cantidad de UCIs disponibles, y a la forma en que la sociedad afronta el distanciamiento social.
    *   Se observa que el modelo de predicción funciona mejor con la tendencia mientras mas datos tenga para entrenar, como sucede en el caso del dataset inicial que esta dividido por días, en cambio el de semana tiene un ajuste mucho mas pobre, debido a la reducción de la cantidad de días.
    *    El modelo de predicción muestra una tendencia alcista para los siguientes 15 días, que puede ser asociada al momento del entrenamiento en que se pasa de pocos casos al día a un  incremento significativo (alrededor del día 90 en los datos reales). Esto puede atribuirse en la realidad a la reapertura económica que nos encontramos viviendo, que aunque necesaria en terminos económicos, va a crear un incremento en la cantidad de infecatdos y fallecidos nuevamente.
    """)