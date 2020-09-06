import pandas as pd
from sodapy import Socrata
import streamlit as st

@st.cache
def load_api():
    #client = Socrata("www.datos.gov.co", None)
    #results = client.get("gt2j-8ykr", limit=1000000)
    client = Socrata("www.datos.gov.co", None)
    #              '5aJ1DExqkSA79jE4S3I27ZMyL',
    #              username="egangiest@gmail.com",
    #              password="5aJ1DExqkSA79jE4S3I27ZMyL")
    results = client.get("gt2j-8ykr", limit=1000000)

    Dataset1 = pd.DataFrame.from_records(results)
    Dataset1.columns = ['ID de caso', 'Fecha de notificación', 'Código DIVIPOLA', 'Ciudad de ubicación',
       'Departamento o Distrito ', 'Atención', 'Edad', 'Sexo', 'Tipo',
       'Estado', 'País de procedencia', 'FIS', 'Fecha de muerte',
       'Fecha diagnostico', 'Fecha recuperado', 'fecha reporte web',
       'Tipo recuperación', 'Codigo departamento', 'Codigo pais',
       'Pertenencia etnica', 'Nombre grupo etnico']
       
    # Limpieza de las columnas con fechas, para eliminar la hora.
    Fechas = ['Fecha de notificación', 'Fecha de muerte' , 'Fecha diagnostico' , 'Fecha recuperado' ]
    for i in Fechas:
        Dataset1[i] = Dataset1[i].apply(lambda x: str(x).split('T')[0])
    # Conversión de las fechas a tipo fecha (datetime de Pandas)
    for i in Fechas:
        Dataset1[i]= pd.to_datetime(Dataset1[i])
    # Cambiando la columna edad de tipo str a tipo int
    Dataset1['Edad'] = Dataset1['Edad'].astype(int)
    # Eliminando columnas que se consideran innecesarias
    Dataset1.drop(columns = ['ID de caso', 'Código DIVIPOLA' , 'FIS','Codigo departamento','fecha reporte web', 
        'Codigo pais' , 'Codigo departamento' , 'Codigo pais','Pertenencia etnica', 'Nombre grupo etnico',
        'Departamento o Distrito ' ] , inplace = True)
    # Ordenando el dataset segun la fecha de notificación
    Dataset1 = Dataset1.sort_values(by=['Fecha de notificación']) 
 
    # Se hace una revisión de los valores únicos de las siguientes columnas
    Listacolumnas = ['Sexo', 'Atención' , 'Tipo', 'Estado' ]
    #for name in Listacolumnas:
        #print("Valores únicos" , name ,  Dataset1[name].unique() )
    # Corrección de minusculas de columna Sexo
    Dataset1['Sexo'] = Dataset1['Sexo'].apply(lambda x: x.upper())
    # Corrección de minusculas de columna Tipo
    Dataset1['Tipo'] = Dataset1['Tipo'].apply(lambda x: x.upper())

    # Función para definir categorias segun el rango de edad
    def categoria_edad(edad):
        if edad <= 11:
            return('Niño')
        if 11 < edad <= 18:
            return('Adolescente')
        if 18 < edad <= 26:        
            return('Juventud')
        if 26 < edad <= 59:        
            return('Adulto')
        else:
            return('Vejez')
    # Creación de columna de categorias de edad
    Dataset1['Cat_Edad'] = Dataset1['Edad'].apply(categoria_edad)

    # Creación de un Subset asociado a las cinco ciudades principales
    Principales = ['Bogotá D.C.', 'Medellín' , 'Cali' , 'Barranquilla' , 'Cartagena de Indias' ]
    Dataset_ciudades = Dataset1[Dataset1['Ciudad de ubicación'].isin(Principales)]

    return Dataset_ciudades




