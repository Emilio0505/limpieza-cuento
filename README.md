# Limpieza de un cuento con PLN

Proyecto sencillo que carga `cuento.txt` y realiza limpieza, lematizacion y vectorizacion con Bag of Words, TF-IDF y Word2Vec.

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
python semantica_distribucional.py
```

El proyecto genera el texto limpio, las matrices de Bag of Words y TF-IDF, un modelo Word2Vec, los vectores semanticos y una grafica 3D.
