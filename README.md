# Limpieza de un cuento con PLN

Proyecto sencillo que carga `cuento.txt` y aplica tokenizacion, eliminacion de palabras vacias, eliminacion de puntuacion, conversion a minusculas y lematizacion. Tambien convierte el texto a numeros usando Bag of Words y TF-IDF.

## Crear el entorno virtual en Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

## Instalar lo necesario

```bash
python -m pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Ejecutar

```bash
python limpieza.py
python vectorizacion.py
```

Al terminar se crean `resultado_limpio.txt`, `resultado_bow.csv` y `resultado_tfidf.csv`.
