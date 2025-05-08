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
