import io

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from bitflip import probability_curve, simulate_bit_flips


def _example_image() -> Image.Image:
    width, height = 320, 220
    gradient_x = np.linspace(0, 255, width, dtype=np.uint8)
    gradient_y = np.linspace(0, 255, height, dtype=np.uint8)
    r = np.tile(gradient_x, (height, 1))
    g = np.tile(gradient_y.reshape(-1, 1), (1, width))
    b = np.full((height, width), 180, dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _load_image(file_buffer: io.BytesIO | None) -> Image.Image:
    if file_buffer is None:
        return _example_image()
    return Image.open(file_buffer).convert("RGB")


def main() -> None:
    st.set_page_config(page_title="bitFlip - Rayos cósmicos", layout="wide")
    st.title("Simulador de bit flips por rayos cósmicos")
    st.write(
        "Explora cómo cambian las tasas de error al modificar altitud, tiempo de exposición y el tamaño en bits de una imagen."
    )

    with st.sidebar:
        st.header("Parámetros de la simulación")
        base_rate = st.number_input(
            "Tasa base por bit (errores/bit·hora)",
            min_value=1e-14,
            max_value=1e-6,
            value=1e-12,
            format="%.1e",
        )
        altitude = st.slider("Altitud [m]", min_value=0, max_value=12000, value=3000, step=250)
        hours = st.slider("Tiempo simulado [h]", min_value=0.01, max_value=24.0, value=1.0, step=0.01)
        seed_option = st.checkbox("Fijar semilla aleatoria", value=False)
        seed_value = st.number_input("Semilla", value=42, step=1, disabled=not seed_option)

        uploaded_file = st.file_uploader("Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"])

    image = _load_image(uploaded_file)
    seed = int(seed_value) if seed_option else None

    result = simulate_bit_flips(
        image=image,
        altitude_m=float(altitude),
        hours=float(hours),
        base_rate_per_bit=float(base_rate),
        seed=seed,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen original")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Imagen corrompida")
        st.image(result.corrupted_image, use_column_width=True)

    st.markdown("---")
    st.subheader("Resultados")
    st.write(
        "La tasa se escala con la altitud como $2^{h/1000}$ y se modela con un proceso de Poisson para los errores."
    )
    st.metric("Bits totales", f"{result.total_bits:,}")
    st.metric("Errores esperados (μ)", f"{result.expected_errors:.3e}")
    st.metric("Errores observados", f"{result.observed_errors}")
    st.metric("Prob. ≥ 1 error", f"{result.probability_at_least_one:.2%}")

    st.subheader("Probabilidad acumulada vs tiempo")
    hours_range = np.linspace(0, max(hours, 12), 40)
    probabilities = probability_curve(
        base_rate_per_bit=float(base_rate),
        altitude_m=float(altitude),
        num_bits=result.total_bits,
        hours_points=hours_range,
    )
    df = pd.DataFrame({"Horas": hours_range, "Probabilidad": probabilities})
    st.line_chart(df, x="Horas", y="Probabilidad")

    st.caption(
        "Prototipo educativo: ajusta los parámetros y observa cómo aumenta la probabilidad de fallos y los artefactos visuales."
    )


if __name__ == "__main__":
    main()
