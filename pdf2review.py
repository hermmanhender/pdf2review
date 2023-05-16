import os
import PyPDF2
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gpt import GPT, Example
import pydot

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Descargar recursos adicionales de NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Ruta de la carpeta que contiene los archivos PDF
carpeta_pdf = 'ruta/a/la/carpeta'

# Lista de subtemas
subtemas = ['Subtema 1', 'Subtema 2', 'Subtema 3', 'Subtema 4', 'Subtema 5']

# Configuración de GPT
gpt = GPT(engine="text-davinci-003")

# Cargar modelo de GPT preentrenado
gpt.add_example(Example('What is the main idea of this article?', 'The main idea of this article is...'))

def obtener_metadatos_pdf(filepath):
    # Solicitar al usuario ingresar los metadatos del PDF
    title = input(f"Ingrese el título del archivo '{filepath}': ")
    author = input(f"Ingrese el autor del archivo '{filepath}': ")
    year = input(f"Ingrese el año de publicación del archivo '{filepath}': ")
    
    metadatos = {
        'title': title,
        'author': author,
        'year': year
    }
    
    return metadatos

def leer_contenido_pdf(filepath):
    with open(filepath, 'rb') as f:
        pdf = PyPDF2.PdfFileReader(f)
        num_pages = pdf.getNumPages()
        
        contenido = ''
        for page_num in range(num_pages):
            page = pdf.getPage(page_num)
            contenido += page.extractText()
        
    return contenido

def clasificar_subtema(text, subtemas):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens if token.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]

    # Buscar subtemas existentes similares
    similitud_minima = 0.8  # Umbral de similitud mínima
    subtema_existente = None

    for subtema in subtemas:
        subtema_tokens = word_tokenize(subtema)
        subtema_tokens = [token.lower() for token in subtema_tokens if token.isalpha()]
        subtema_stemmed_tokens = [stemmer.stem(token) for token in subtema_tokens]

        # Calcular similitud entre los tokens del subtema existente y los tokens del texto
        similitud = len(set(stemmed_tokens).intersection(subtema_stemmed_tokens)) / len(set(stemmed_tokens))

        if similitud >= similitud_minima:
            subtema_existente = subtema
            break

    if subtema_existente is None:
        # Si no se encontró un subtema existente similar, se asigna uno nuevo
        subtema_nuevo = f"Subtema {len(subtemas) + 1}"
        subtemas.append(subtema_nuevo)
        return subtema_nuevo
    else:
        # Si se encontró un subtema existente similar, se utiliza ese
        return subtema_existente

def solicitar_notas_clave(contenido):
    # Dividir el contenido en oraciones
    oraciones = nltk.sent_tokenize(contenido)
    
    # Calcular la matriz de características utilizando TF-IDF
    vectorizer = TfidfVectorizer()
    matriz_caracteristicas = vectorizer.fit_transform(oraciones)
    
    # Calcular la matriz de similitud entre las oraciones
    matriz_similitud = cosine_similarity(matriz_caracteristicas)
    
    # Calcular la similitud promedio para cada oración
    similitud_promedio = np.mean(matriz_similitud, axis=1)
    
    # Seleccionar las 3 oraciones más relevantes como notas clave
    indices_notas_clave = np.argsort(similitud_promedio)[-3:]
    notas_clave = [oraciones[i] for i in indices_notas_clave]
    
    return notas_clave

def es_pdf_relevante(contenido_pdf, resumen_articulo):
    # Implementa tu lógica de filtrado aquí
    # Puedes comparar el contenido del PDF con el resumen del artículo utilizando técnicas de procesamiento de lenguaje natural, como similitud de coseno, coincidencia de palabras clave, etc.
    # Aquí te dejo un ejemplo simple basado en la similitud de coseno
    
    # Obtener el contenido del resumen del artículo en oraciones
    oraciones_resumen = nltk.sent_tokenize(resumen_articulo)
    
    # Obtener el contenido del PDF en oraciones
    oraciones_pdf = nltk.sent_tokenize(contenido_pdf)
    
    # Calcular la matriz de características utilizando TF-IDF
    vectorizer = TfidfVectorizer()
    matriz_caracteristicas = vectorizer.fit_transform(oraciones_resumen + oraciones_pdf)
    
    # Calcular la similitud de coseno entre el resumen y el contenido del PDF
    similitud_coseno = cosine_similarity(matriz_caracteristicas[-len(oraciones_pdf):], matriz_caracteristicas[:-len(oraciones_pdf)])
    
    # Calcular el promedio de similitud de coseno
    similitud_promedio = np.mean(similitud_coseno)
    
    # Determinar si el PDF es relevante según un umbral de similitud
    umbral_similitud = 0.5  # Ajusta este umbral según tus necesidades
    if similitud_promedio >= umbral_similitud:
        return True
    
    return False

def filtrar_pdfs_relevantes(archivos_pdf, resumen_articulo):
    pdfs_relevantes = []
    
    for archivo_pdf in archivos_pdf:
        contenido_pdf = leer_contenido_pdf(archivo_pdf)
        # Aquí puedes realizar la lógica de filtrado para determinar si el PDF es relevante para el resumen del artículo
        # Puedes comparar el contenido del PDF con el resumen del artículo o utilizar otras técnicas de procesamiento de lenguaje natural
        
        if es_pdf_relevante(contenido_pdf, resumen_articulo):
            pdfs_relevantes.append(archivo_pdf)
    
    return pdfs_relevantes

def generar_resumen(folder_path, resumen_articulo):
    # Obtener la lista de archivos PDF en la carpeta de origen
    archivos_pdf = leer_contenido_pdf(folder_path)
    
    # Filtrar los PDFs relevantes para el resumen del artículo
    pdfs_relevantes = filtrar_pdfs_relevantes(archivos_pdf, resumen_articulo)
    
    # Crear un nuevo documento Word
    doc = Document()
    
    # Agregar el resumen del artículo al documento
    doc.add_heading('Resumen del artículo', level=1)
    doc.add_paragraph(resumen_articulo)
    
    # Agregar las citas de los PDFs relevantes al documento
    doc.add_heading('Citas de los PDFs relevantes', level=1)
    for pdf_relevante in pdfs_relevantes:
        doc.add_paragraph(f"Cita del PDF: {pdf_relevante}")
    
    # Guardar el documento en formato .docx
    doc.save('resumen_articulo.docx')
    
    print('Se ha generado el resumen del artículo y se ha guardado en un documento Word.')
    
# Procesar archivos PDF
datos_bibliograficos = []
notas_clave = []

for filename in os.listdir(carpeta_pdf):
    if filename.endswith('.pdf'):
        filepath = os.path.join(carpeta_pdf, filename)
        
        metadatos = obtener_metadatos_pdf(filepath)
        contenido = leer_contenido_pdf(filepath)
        subtema = clasificar_subtema(contenido)
        notas_clave = solicitar_notas_clave(filename)
        resumen = generar_resumen(contenido)
        
        # Agregar datos bibliográficos a la lista
        datos_bibliograficos.append({
            'Title': metadatos['title'],
            'Author': metadatos['author'],
            'Year': metadatos['year'],
            'Subtema': subtema,
            'Abstract': resumen,
        })

# Crear DataFrame de pandas con los datos bibliográficos y notas clave
df = pd.DataFrame(datos_bibliograficos)
df['Notas clave'] = notas_clave

# Guardar DataFrame en un archivo Excel
df.to_excel('revision_bibliografica.xlsx', index=False)
