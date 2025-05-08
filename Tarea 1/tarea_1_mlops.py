"""
##Importar librerías
Primero vamos a importar librerías: Numpy, pandas, spacy, matplot y otras para el análisis de datos.

Iniciamos intalando "spacy", usando unos comandos un poco diferentes a los import, debido a que no es una biblioteca "común" para usar dentro de Colab.
Luego, importamos las librerías mencionadas anteriormente, realizamos la carga de datos de los reviews de la película "El señor de los Anillos: La comunidad del anillo" y analizamos un poco la información contenida.
"""

!pip install -q spacy transformers
!python -m spacy download en_core_web_sm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
import pandas as pd
from spacy import displacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
import string
string.punctuation
from collections import defaultdict
from sklearn.metrics import confusion_matrix

plt.style.use('seaborn-v0_8-whitegrid')


df=pd.read_csv('~/The Lord of the Rings - The Fellowship of the Ring - Reviews.csv',encoding = "ISO-8859-1")
print(df.head())

df.info()

df.isna().sum()

df['Rating'].value_counts()

y=df['Rating'].values
y.shape

x=df['Description'].values

x.shape

"""##Pre-Procesamiento de Datos
En este paso, haremos una limpieza de datos en la parte de simbolos y signos de puntuación. Esta primer función, "remove_punctuation" lo que hace es buscar signos de puntuación dentro del texto y eliminarlos.

"generate-n-grams", además de ayudarnos con los n-grams, nos ayuda a limpiar los stop words que tengamos en el texto, usado para varios de los pasos mas abajo.
"""

#defining the function to remove punctuation
def remove_punctuation(text):
  if(type(text)==float):
    return text
  ans=""
  for i in text:
    if i not in string.punctuation:
      ans+=i
  return ans

#storing the puntuation free text in a new column called clean_msg
df=df.assign(DescriptionWOPunctuation=df['Description'].apply(lambda x:remove_punctuation(x)))

print(df.head())
#punctuations are removed from news column in train dataset

"""Aqui podemos ver en el primer elemento de la lista tuvimos K-retailer y ahora se muestra como kretailer. eliminamos el "-"
"""

def generate_N_grams(text,ngram):
  words=[word for word in text.split(" ") if word not in set(stopwords.words('english'))]
  print("Sentence after removing stopwords:",words)
  temp=zip(*[words[i:] for i in range(0,ngram)])
  ans=[' '.join(ngram) for ngram in temp]
  return ans