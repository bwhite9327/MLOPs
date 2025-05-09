# MLOPs - Tarea 1

## Tarea 1
 Modelo básico de ML y Github
### Contexto:
    Esta tarea, busca practicar el uso de un programa de cambios en código para un programa en desarrollos corporativos.
    El contexto, es una "una pequeña tienda en línea quiere automatizar la clasificación de reseñas de client (positivas o negativas). Te han pedido crear un modelo simple de análisis de sentimientos."
 
### Instrucciones:
    - Usa un dataset pequeño de comentarios (puedes usar uno propio o alguno como el de Sentiment140, o reseñas de IMDb).
    - Entrena un modelo simple de clasificación binaria (positivo/negativo) usando scikit-learn.
    - Sube el proyecto a un repositorio en GitHub:
         - Crea un branch para trabajar en el.
         - Incluye al menos 3 commits bien redactados.
         - README breve explicando el objetivo y cómo ejecutar el código

### Cómo ejecutar el código?
   Este proyecto, requiere clonar el repositorio a una carpeta destino, y ajustar la ubicación del archivo de las reseñas según la ubicación en la computadora destino.
   Una vez realizado esto, el código tiene 3 partes:
      - Importación de librerías
      - Pre-Procesamiento de Datos
      - Análisis de sentimiento

   En la primera parte, vamos a importar librerías: Numpy, pandas, spacy, matplot y otras para el análisis de datos.
   Iniciamos intalando "spacy" y luego importamos las librerías mencionadas anteriormente.
   
   Para el pre-procesamiento de datos, realizamos la carga de datos de los reviews de la película "El señor de los Anillos: La comunidad del anillo" y analizamos un poco la información contenida.
   
   Haremos una limpieza de datos en la parte de simbolos y signos de puntuación. La primer función, "remove_punctuation" lo que hace es buscar signos de puntuación dentro del texto y eliminarlos.
   "generate-n-grams", además de ayudarnos con los n-grams, nos ayuda a limpiar los stop words que tengamos en el texto, usado para varios de los pasos mas abajo.

   Para la parte final, utilizaremos `TfidfVectorizer` para vectorizar los tweets. Se utilizará el parámetro *max_features*, para limitar la cantidad de términos a evaluar. Se vectorizan los tweets, ya que queremos normalizar lo que el sistema usa de input como su fuente de información. Con las palabras y mensajes usados en los tweets, podemos empezar a predecir si el review se considera positivo o negativo.

   Finalmente, utilizaremos los diferentes algoritmos para estimar el sentimiento según el contenido de las oraciones. Los resultados, los sumaremos al DF *dfWorkingSentiments* para preservar el detalle del resultado.

   Este puede ser 0 o 1 si es negativo, o positivo.