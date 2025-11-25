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

## Ejemplos incluidos

- **Memoria como imagen**: simula bit flips sobre una imagen RGB, con animación temporal y curvas de probabilidad que siguen un modelo de Poisson y escalado exponencial con la altitud ($2^{h/1000}$).
- **Biosimulación de ADN**: pestaña alternativa que trata cada base (A/C/G/T) como un "bit" susceptible a mutar. Puedes fijar altitud, horas, número de frames, semilla y generar secuencias aleatorias. La secuencia mutada se muestra con las bases alteradas resaltadas para visualizar el avance de las mutaciones.
