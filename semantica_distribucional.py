from pathlib import Path
import spacy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as grafica
from gensim.models import Word2Vec as PalabrasAVectores
from sklearn.decomposition import PCA as ComponentesPrincipales

carpeta = Path(__file__).parent
nlp = spacy.load("es_core_news_sm")

with open(carpeta / "cuento.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()

documento = nlp(texto)
oraciones = []

for oracion in documento.sents:
    palabras = []

    for palabra in oracion:
        if not palabra.is_stop and not palabra.is_punct and not palabra.is_space:
            palabras.append(palabra.lemma_.lower())

    if len(palabras) > 1:
        oraciones.append(palabras)

modelo = PalabrasAVectores(
    oraciones,
    vector_size=10,
    window=3,
    min_count=1,
    workers=1,
    seed=40,
    epochs=200
)

modelo.save(str(carpeta / "modelo_word2vec.model"))

vocabulario = list(modelo.wv.index_to_key)
vectores = modelo.wv[vocabulario]

with open(carpeta / "vectores_semanticos.txt", "w", encoding="utf-8") as archivo:
    for palabra in vocabulario:
        valores = " ".join(str(round(valor, 4)) for valor in modelo.wv[palabra])
        archivo.write(palabra + ": " + valores + "\n")

reduccion = ComponentesPrincipales(n_components=3)
vectores_3d = reduccion.fit_transform(vectores)

figura = grafica.figure(figsize=(12, 8))
ejes = figura.add_subplot(111, projection="3d")
ejes.scatter(vectores_3d[:, 0], vectores_3d[:, 1], vectores_3d[:, 2], color="crimson")

for numero, palabra in enumerate(vocabulario):
    ejes.text(vectores_3d[numero, 0], vectores_3d[numero, 1], vectores_3d[numero, 2], palabra)

ejes.set_title("Espacio semantico del cuento")
ejes.set_xlabel("Dimension 1")
ejes.set_ylabel("Dimension 2")
ejes.set_zlabel("Dimension 3")
grafica.tight_layout()
grafica.savefig(carpeta / "grafica_semantica.png", dpi=150)

print("Oraciones procesadas:", len(oraciones))
print("Palabras en el vocabulario:", len(vocabulario))

for palabra in ["hormiga", "luciérnaga"]:
    if palabra in modelo.wv:
        print("\nPalabras similares a", palabra)
        for similar, valor in modelo.wv.most_similar(palabra, topn=3):
            print(similar, round(valor, 4))

print("\nSe crearon el modelo los vectores y la grafica")
