# bitFlip

Prototipo educativo para simular cómo los rayos cósmicos pueden provocar **bit flips** en memoria y visualizar el efecto sobre una imagen. Usa Streamlit para ajustar parámetros como altitud, tiempo de exposición, tasa base de errores y reproducir una animación que muestra la corrupción progresiva.

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

Después abre la URL que te indique la consola (por defecto http://localhost:8501). Puedes subir tu propia imagen o usar alguno de los ejemplos (gradiente, cuadrícula binaria o texto). Ajusta altitud, tiempo, número de frames y tasa base; luego reproduce la animación o usa el selector temporal para ver cómo se va corrompiendo la imagen y cómo crece la probabilidad de fallo.
