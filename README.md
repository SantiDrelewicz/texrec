# texrec
Predictor de capitalización y Capitalización de Texto Normalizado

## Motivación
Este tipo de sistema tiene aplicaciones reales en procesamiento de lenguaje natural. Un caso típico es el
postprocesamiento de la salida de sistemas de reconocimiento automático del habla (ASR, por sus siglas
en inglés). Estos sistemas suelen producir texto plano, en minúsculas y sin puntuación, lo cual dificulta su
lectura y comprensión.
El objetivo de este modelo es "reconstruir" ese texto enriquecido con
puntuación y mayúsculas, mejorando su utilidad en tareas posteriores como: Resumen automático,
Análisis de sentimiento o Traducción automática.

📄 Ejemplo

- **Entrada original:**
  “cuándo vamos a mcdonald's ellos no vienen hoy dónde están ahora”

- **Texto esperado reconstruido:**
  "¿Cuándo vamos a McDonald's? Ellos no vienen hoy. ¿Dónde están ahora?"


