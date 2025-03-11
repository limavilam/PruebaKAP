import spacy
import pandas as pd
from spacy.tokens import Doc
from spacy.lang.es import Spanish

nlp = spacy.load("es_core_news_sm")

def limpieza_normalizar(texto):
    caracteres_filtrado=[]
    for caracter in texto:
        if caracter.isalpha() or caracter.isspace():
            caracteres_filtrado.append(caracter)
    texto= ''.join(caracteres_filtrado)
    texto = texto.lower()
    return texto

#texto = "¡Hola, Mundo! 123. ¿Cómo estás?"
#texto_limpio = limpieza_normalizar(texto)
#print(texto_limpio)

def procesamiento_texto(texto):
    texto_limpio = limpieza_normalizar(texto)
    doc = nlp(texto_limpio)
    oraciones_con_entidades = []
    
    for sent in doc.sents:
        entities = []
        entity_types = []

        for ent in sent.ents:
            if ent.label_ in ["PER", "ORG", "LOC"]:
                entities.append(ent.text)
                entity_types.append(ent.label_)
        
        if entities:  
            oraciones_con_entidades.append({
                "oración": sent.text,
                "entidades encontradas": entities,
                "tipo de entidad": entity_types
            })

    return oraciones_con_entidades

#texto = "Londres está ubicado en el Reino Unido"
#resultado = procesamiento_texto(texto)
#print(resultado)

def procesar_articulos(articulos):
    resultados = []
    for articulo in articulos:
        resultados.extend(procesamiento_texto(articulo))
    df = pd.DataFrame(resultados)
    return df

arts = [
    
    "La ONU anunció una nueva iniciativa en colaboración con el gobierno de Colombia.",
    "El presidente de Apple, anuncio una nueva oficina en Seattle",
    "El equipo de fútbol de Colombia ganó el mundial."
]

df = procesar_articulos(arts)
df.to_csv("Entidades_extraidas.csv", index=False)

print(df)

"""
Evaluación de calidad de extracción:

Podría utilizar las métricas y validaciones como:
- Precision
- Recall
- F1-Score

"""