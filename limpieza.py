from pathlib import Path
import spacy

nlp = spacy.load("es_core_news_sm")

carpeta = Path(__file__).parent

with open(carpeta / "cuento.txt", "r", encoding="utf-8") as archivo:
    texto = archivo.read()
documento = nlp(texto)
tokens_originales = [token.text for token in documento]
tokens_limpios = []
for token in documento:
    if not token.is_stop and not token.is_punct and not token.is_space:
        tokens_limpios.append(token.lemma_.lower())

texto_limpio = " ".join(tokens_limpios)

with open(carpeta / "resultado_limpio.txt", "w", encoding="utf-8") as archivo:
    archivo.write(texto_limpio)

print("Primeros 20 tokens originales:")
print(tokens_originales[:20])
print("\nPrimeros 20 tokens limpios y lematizados:")
print(tokens_limpios[:20])
print("\nTokens originales:", len(tokens_originales))
print("Tokens despues de la limpieza:", len(tokens_limpios))
print("Se creo resultado_limpio.txt")
