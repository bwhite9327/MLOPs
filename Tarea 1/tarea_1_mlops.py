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

"""#Analisis de sentimiento
* **sentiment140-subset.csv:** es un subset de Sentiment140 data - que proviene de medio million de tweets marcados como positivos o negativos.
"""

# Make data directory if it doesn't exist
!mkdir -p data
!wget -nc https://nyc3.digitaloceanspaces.com/ml-files-distro/v1/investigating-sentiment-analysis/data/sentiment140-subset.csv.zip -P data
!unzip -n -d data data/sentiment140-subset.csv.zip

# !pip install sklearn

dfSentiments = pd.read_csv("data/sentiment140-subset.csv", nrows=30000)
print(dfSentiments.head())

"""## Vectorizar los tweets

Utilizaremos `TfidfVectorizer` para vectorizar los tweets. Se utilizará el parámetro *max_features*, para limitar la cantidad de términos a evaluar.
"""

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=1000)
vectors = vectorizer.fit_transform(dfSentiments.text)
words_df = pd.DataFrame(vectors.toarray(), columns=vectorizer.get_feature_names_out())
words_df.head()

"""### Configurar las variables

`X` son todas las características para predecir si es positivo o negativo. En este caso, serán las palabras.

`y` son todas las puntuaciones de etiquetas, positivas o negativass. La polaridad nos va a ayudar a determinar eso.
"""

X = words_df
y = dfSentiments.polarity

"""### Algoritmo de predicción

Ya que estamos aprovechando un ejercicio pasado, usaremos todos los algoritmos que venían y veremos como se diferencia uno de otro.
"""

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB

"""### Entrenar los algoritmos

En los siguientes pasos, utilizaremos los ejes X (el analisis de palabras de twitter) y Y (la polaridad que tienen las palabras) en los diferentes algoritmos:

* Regresión Logaritmica
* Clasificador Boque Aleatorio
* Clasificador de vector de soporte lineal
* Clasificador multinomial de Bayes ingenuo
"""

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Create and train a logistic regression
# logreg = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
# logreg.fit(X, y)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Create and train a random forest classifier
# forest = RandomForestClassifier(n_estimators=50)
# forest.fit(X, y)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Create and train a linear support vector classifier (LinearSVC)
# svc = LinearSVC()
# svc.fit(X, y)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# # Create and train a multinomial naive bayes classifier (MultinomialNB)
# bayes = MultinomialNB()
# bayes.fit(X, y)

"""## Usando los modelos

Ahora que está entrenado el modelo con la info de Twitter, podemos empezar a predecir si el review se considera positivo o negativo.

### Fuente de información

Crearemos otro DF basado en el mismo archivo de reviews del señor de los Anillos; sin embargo, en el pd.read solamente leeremos la columna de _Description_, ya que es la que nos interesa evaluar si es positiva o negativa
"""

dfWorkingSentiments=pd.read_csv('drive/MyDrive/Colab Notebooks/NLP/Proyecto Final/The Lord of the Rings - The Fellowship of the Ring - Reviews.csv',encoding = "ISO-8859-1",usecols=["Description"])
print(dfWorkingSentiments.head())

"""Ahora, vamos a vectorizar las oraciones en números, para que el algoritmo lo pueda entender.

Debemos tomar en cuenta que nuestro algoritmo solo conoce unas cuantas palabras, que se muestran a continuación:
"""

print(vectorizer.get_feature_names_out())

"""Ya que nosotros sabemos las palabras a evaluar, solo ocupamos contar las palabras en el DF de evaluación, por lo que usaremos `.transform`:

```python
unknown_vectors = vectorizer.transform(unknown.content)
unknown_words_df = ......
```

Crearemos un DF `unknown_words_df` para listar todas aquellas palabras que no conocemos de nuestro source data.
"""

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# 
# print("Training logistic regression")
# logreg.fit(X_train, y_train)
# 
# print("Training random forest")
# forest.fit(X_train, y_train)
# 
# print("Training SVC")
# svc.fit(X_train, y_train)
# 
# print("Training Naive Bayes")
# bayes.fit(X_train, y_train)

# Put it through the vectoriser

# transform, not fit_transform, because we already learned all our words
unknown_vectors = vectorizer.transform(dfWorkingSentiments.Description)
unknown_words_df = pd.DataFrame(unknown_vectors.toarray(), columns=vectorizer.get_feature_names_out())
unknown_words_df.head()

unknown_words_df.shape

"""### Predecir el sentimiento con los algoritmos

Finalmente, utilizaremos los diferentes algoritmos para estimar el sentimiento según el contenido de las oraciones. Los resultados, los sumaremos al DF *dfWorkingSentiments* para preservar el detalle del resultado.

Este puede ser 0 o 1 si es negativo, o positivo.
"""

# Predict using all our models.

# Logistic Regression predictions + probabilities
dfWorkingSentiments['pred_logreg'] = logreg.predict(unknown_words_df)
dfWorkingSentiments['pred_logreg_proba'] = logreg.predict_proba(unknown_words_df)[:,1]

# Random forest predictions + probabilities
dfWorkingSentiments['pred_forest'] = forest.predict(unknown_words_df)
dfWorkingSentiments['pred_forest_proba'] = forest.predict_proba(unknown_words_df)[:,1]

# SVC predictions
dfWorkingSentiments['pred_svc'] = svc.predict(unknown_words_df)

# Bayes predictions + probabilities
dfWorkingSentiments['pred_bayes'] = bayes.predict(unknown_words_df)
dfWorkingSentiments['pred_bayes_proba'] = bayes.predict_proba(unknown_words_df)[:,1]

dfWorkingSentiments

y_true = y_train
y_pred = logreg.predict(X_train)
matrix = confusion_matrix(y_true, y_pred)

label_names = pd.Series(['negative', 'positive'])
pd.DataFrame(matrix,
     columns='Predicted ' + label_names,
     index='Is ' + label_names)

"""## Conclusión

Ya que, listar todos los reviews en un chart es dificil de leer, utilizaremos el identificador unico del review para evaluarlo. A continuación, la lista
"""

print(dfWorkingSentiments.Description)

# prompt: I would love a chart where I list all the y_train from the "Description" dfWorkingSentiments in the Y axis, and for the values, use a line chart that has multiple lines: -pred_logreg, pred_fores, pred-svc and pred-bayes, each line with a different colour

import matplotlib.pyplot as plt

# Assuming dfWorkingSentiments is your DataFrame
# and y_train is your training target variable

plt.figure(figsize=(12, 6))

plt.plot(dfWorkingSentiments.index, dfWorkingSentiments['pred_logreg'], label='Logistic Regression', color='blue')
plt.plot(dfWorkingSentiments.index, dfWorkingSentiments['pred_forest'], label='Random Forest', color='green')
plt.plot(dfWorkingSentiments.index, dfWorkingSentiments['pred_svc'], label='SVC', color='red')
plt.plot(dfWorkingSentiments.index, dfWorkingSentiments['pred_bayes'], label='Naive Bayes', color='orange')

plt.xlabel('Index')
plt.ylabel('y_train from Description')
plt.title('Sentiment Predictions for Descriptions')
plt.legend()
plt.grid(True)

plt.show()

"""Viendo el gráfico, podemos ver como los reviews 9,10,11 y 12 son evaluados como positivos en los 4 algoritmos:

9     Movie Title: The Lord of the Rings: The Fellow...
10    This is one of the biggest triology in history...
11    A must watch for everyone, very entertaining k...
12    The Lord of the Rings is a masterpiece! Peter ...

Viendo el preview que el DF nos arroja, tiene sentido lo que vemos, parecen ser reviews con un tono de escritura positivos. Por el otro lado:

27    After having this movie on my watchlist for ye...
28    Lord of The Rings is, to me, the industry stan...

Nos aparenta un tono de escritura un poco menos positivo, o incluso sentimiento negativo.
"""