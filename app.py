import io

import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from bitflip import probability_curve, simulate_bit_flips_over_time


def _gradient_image() -> Image.Image:
    width, height = 320, 220
    gradient_x = np.linspace(0, 255, width, dtype=np.uint8)
    gradient_y = np.linspace(0, 255, height, dtype=np.uint8)
    r = np.tile(gradient_x, (height, 1))
    g = np.tile(gradient_y.reshape(-1, 1), (1, width))
    b = np.full((height, width), 180, dtype=np.uint8)
    rgb = np.stack([r, g, b], axis=2)
    return Image.fromarray(rgb, mode="RGB")


def _checkerboard_image(size: int = 12) -> Image.Image:
    width, height = 320, 220
    pattern = np.indices((height, width)).sum(axis=0) % 2
    block = np.kron(pattern, np.ones((size, size)))
    block = block[:height, :width]
    rgb = np.stack([
        block * 255,
        (1 - block) * 255,
        np.full_like(block, 128),
    ], axis=2).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def _text_image() -> Image.Image:
    width, height = 320, 220
    img = Image.new("RGB", (width, height), color=(10, 15, 25))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    message = "bitFlip"
    text_bbox = draw.textbbox((0, 0), message, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)
    draw.text(position, message, fill=(200, 240, 255), font=font)
    draw.rectangle([20, 20, width - 20, height - 20], outline=(90, 200, 255), width=2)
    return img


def _example_image(choice: str) -> Image.Image:
    if choice == "Cuadrícula binaria":
        return _checkerboard_image()
    if choice == "Texto 'bitFlip'":
        return _text_image()
    return _gradient_image()


def _load_image(file_buffer: io.BytesIO | None, choice: str) -> Image.Image:
    if file_buffer is None:
        return _example_image(choice)
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
        hours = st.slider("Tiempo total simulado [h]", min_value=0.01, max_value=24.0, value=1.0, step=0.01)
        steps = st.slider("Frames en la animación", min_value=4, max_value=60, value=12)
        seed_option = st.checkbox("Fijar semilla aleatoria", value=False)
        seed_value = st.number_input("Semilla", value=42, step=1, disabled=not seed_option)

        example_choice = st.selectbox(
            "Si no subes una imagen, usa este ejemplo:",
            ["Gradiente suave", "Cuadrícula binaria", "Texto 'bitFlip'"],
        )
        uploaded_file = st.file_uploader("Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"])

    image = _load_image(uploaded_file, example_choice)
    seed = int(seed_value) if seed_option else None

    frames = simulate_bit_flips_over_time(
        image=image,
        altitude_m=float(altitude),
        total_hours=float(hours),
        steps=int(steps),
        base_rate_per_bit=float(base_rate),
        seed=seed,
    )
    frame_labels = [f"{frame.time_hours:.2f} h" for frame in frames]
    selected_label = st.select_slider(
        "Explora cómo se va corrompiendo la imagen con el tiempo",
        options=frame_labels,
        value=frame_labels[-1],
    )
    selected_index = frame_labels.index(selected_label)
    current_frame = frames[selected_index]
    latest_frame = frames[-1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Imagen original")
        st.image(image, use_column_width=True)
    with col2:
        st.subheader("Imagen corrompida / animación")
        anim_placeholder = st.empty()
        anim_placeholder.image(current_frame.image, use_column_width=True, caption=f"t = {current_frame.time_hours:.2f} h")

        if st.button("Reproducir animación", type="primary"):
            for frame in frames:
                anim_placeholder.image(
                    frame.image,
                    use_column_width=True,
                    caption=f"t = {frame.time_hours:.2f} h — errores acumulados: {frame.observed_errors}",
                )
                time.sleep(0.35)

    st.markdown("---")
    st.subheader("Resultados")
    st.write(
        "La tasa se escala con la altitud como $2^{h/1000}$ y se modela con un proceso de Poisson para los errores."
    )
    rgb_image = image.convert("RGB")
    bits_total = rgb_image.size[0] * rgb_image.size[1] * 3 * 8
    st.metric("Bits totales", f"{bits_total:,}")
    st.metric("Errores esperados (μ)", f"{latest_frame.expected_errors:.3e}")
    st.metric("Errores observados", f"{latest_frame.observed_errors}")
    st.metric("Prob. ≥ 1 error", f"{latest_frame.probability_at_least_one:.2%}")

    st.subheader("Probabilidad acumulada vs tiempo")
    hours_range = np.linspace(0, max(hours, 12), 40)
    probabilities = probability_curve(
        base_rate_per_bit=float(base_rate),
        altitude_m=float(altitude),
        num_bits=bits_total,
        hours_points=hours_range,
    )
    df = pd.DataFrame({"Horas": hours_range, "Probabilidad": probabilities})
    st.line_chart(df, x="Horas", y="Probabilidad")

    st.caption(
        "Prototipo educativo: ajusta los parámetros y observa cómo aumenta la probabilidad de fallos y los artefactos visuales."
    )

    st.markdown("---")
    st.subheader("Ideas visuales para hablar de rayos cósmicos")
    st.write(
        "Experimenta también con:"
    )
    st.markdown(
        """
        - **SRAM tipo caché**: usa la cuadrícula binaria para ilustrar celdas alternas y ver cómo saltan los errores.
        - **Textos o logos**: elige el ejemplo de texto para mostrar cómo aparecen artefactos incluso en mensajes cortos.
        - **Imágenes en escala de grises o mapas térmicos**: sube una imagen de sensores para visualizar cómo un único bit altera mediciones.
        - **Animación de fallos acumulados**: reproduce la animación para comparar oficina vs avión vs órbita baja cambiando la altitud.
        """
    )


if __name__ == "__main__":
    main()
