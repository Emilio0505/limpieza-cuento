from pathlib import Path
import csv
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Cargar el modelo de espanol
nlp = spacy.load("es_core_news_sm")
carpeta = Path(__file__).parent

# Leer el cuento
with open(carpeta / "cuento.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

# Cada parrafo sera un documento
documentos = [parrafo.strip() for parrafo in texto.split("\n") if parrafo.strip()]


def limpiar(texto):
    documento = nlp(texto)
    palabras = []

    for token in documento:
        if not token.is_stop and not token.is_punct and not token.is_space:
            palabras.append(token.lemma_.lower())

    return palabras


def guardar_csv(nombre, matriz, palabras):
    with open(carpeta / nombre, "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(["documento"] + list(palabras))

        for numero, fila in enumerate(matriz, start=1):
            escritor.writerow([numero] + list(fila))


# Bag of Words cuenta cuantas veces aparece cada palabra
vectorizador_bow = CountVectorizer(analyzer=limpiar)
matriz_bow = vectorizador_bow.fit_transform(documentos)
palabras_bow = vectorizador_bow.get_feature_names_out()

# TF-IDF calcula la importancia de cada palabra
vectorizador_tfidf = TfidfVectorizer(analyzer=limpiar)
matriz_tfidf = vectorizador_tfidf.fit_transform(documentos)
palabras_tfidf = vectorizador_tfidf.get_feature_names_out()

# Guardar las matrices para poder revisarlas
guardar_csv("resultado_bow.csv", matriz_bow.toarray(), palabras_bow)
guardar_csv("resultado_tfidf.csv", matriz_tfidf.toarray().round(3), palabras_tfidf)

print("Documentos procesados:", len(documentos))
print("Palabras encontradas:", len(palabras_bow))
print("\nPrimeras palabras del vocabulario:")
print(palabras_bow[:20])
print("\nVector Bag of Words del primer documento:")
print(matriz_bow.toarray()[0])
print("\nVector TF-IDF del primer documento:")
print(matriz_tfidf.toarray()[0].round(3))
print("\nSe crearon resultado_bow.csv y resultado_tfidf.csv")
