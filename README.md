# bitFlip

Prototipo educativo para simular cómo los rayos cósmicos pueden provocar **bit flips** en memoria y visualizar el efecto sobre una imagen. Usa Streamlit para ajustar parámetros como altitud, tiempo de exposición y tasa base de errores.

## Requisitos

- Python 3.11+
- Dependencias en `requirements.txt`

Instala las dependencias con:

```bash
pip install -r requirements.txt
```

## Ejecución del prototipo

Inicia la aplicación local de Streamlit:

```bash
streamlit run app.py
```

Después abre la URL que te indique la consola (por defecto http://localhost:8501). Puedes subir tu propia imagen o usar el gradiente de ejemplo. Ajusta altitud, tiempo y tasa base para observar las probabilidades de fallo y cómo se corrompe la imagen.
