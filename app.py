import io

import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

from bitflip import (
    mutation_probability_curve,
    probability_curve,
    simulate_bit_flips_over_time,
    simulate_dna_mutations_over_time,
)
from bitflip.dna_simulation import DNA_BASES


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


def _wrap_sequence(seq: str, width: int = 60) -> list[str]:
    return [seq[i : i + width] for i in range(0, len(seq), width)]


def _highlight_sequence(sequence: str, reference: str, width: int = 60) -> str:
    """Return HTML with bases that changed highlighted in yellow."""

    highlighted_lines: list[str] = []
    for seq_chunk, ref_chunk in zip(_wrap_sequence(sequence, width), _wrap_sequence(reference, width)):
        line_parts: list[str] = []
        for base, ref_base in zip(seq_chunk, ref_chunk):
            if base == ref_base:
                line_parts.append(base)
            else:
                line_parts.append(
                    f"<span style='background-color:#ffef9f;color:#0f172a;padding:1px 2px;border-radius:3px;font-weight:600;'>{base}</span>"
                )
        highlighted_lines.append("".join(line_parts))
    return "<br>".join(highlighted_lines)


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
        "cómo cambian las tasas de error al modificar altitud, tiempo de exposición y el tamaño en bits de una imagen."
    )

    default_dna_sequence = (
        "ATGCGTACGTACGTTAGCGTACGATGCTAGCGGATCATCGGCTAGCATGCTACG"
        "TACGGATCGAACGTTAGCAGCTAGCATCGATGCTACGTAGCA"
    )
    if "dna_sequence" not in st.session_state:
        st.session_state["dna_sequence"] = default_dna_sequence

    with st.sidebar:
        st.header("Parámetros de la simulación")
        with st.expander("Imagen / memoria", expanded=True):
            base_rate = st.number_input(
                "Tasa base por bit (errores/bit·hora)",
                min_value=1e-14,
                max_value=1e-6,
                value=1e-12,
                format="%.1e",
                key="img_base_rate",
            )
            altitude = st.slider(
                "Altitud [m]", min_value=0, max_value=12000, value=3000, step=250, key="img_altitude"
            )
            hours = st.slider(
                "Tiempo total simulado [h]", min_value=0.01, max_value=24.0, value=1.0, step=0.01, key="img_hours"
            )
            steps = st.slider("Frames en la animación", min_value=4, max_value=60, value=12, key="img_steps")
            seed_option = st.checkbox("Fijar semilla aleatoria", value=False, key="img_seed_option")
            seed_value = st.number_input(
                "Semilla", value=42, step=1, disabled=not seed_option, key="img_seed_value"
            )

            example_choice = st.selectbox(
                "Si no subes una imagen, usa este ejemplo:",
                ["Gradiente suave", "Cuadrícula binaria", "Texto 'bitFlip'"],
                key="img_example",
            )
            uploaded_file = st.file_uploader(
                "Sube una imagen (PNG/JPG)", type=["png", "jpg", "jpeg"], key="img_uploader"
            )

        with st.expander("ADN biosimulation", expanded=False):
            dna_base_rate = st.number_input(
                "Tasa base por base (mutaciones/base·hora)",
                min_value=1e-14,
                max_value=1e-6,
                value=5e-12,
                format="%.1e",
                key="dna_base_rate",
            )
            dna_altitude = st.slider(
                "Altitud [m] (ADN)", min_value=0, max_value=20000, value=10000, step=250, key="dna_altitude"
            )
            dna_hours = st.slider(
                "Tiempo total simulado [h] (ADN)",
                min_value=0.05,
                max_value=72.0,
                value=4.0,
                step=0.05,
                key="dna_hours",
            )
            dna_steps = st.slider("Frames en la animación (ADN)", min_value=4, max_value=80, value=20, key="dna_steps")
            dna_seed_option = st.checkbox("Fijar semilla aleatoria (ADN)", value=False, key="dna_seed_option")
            dna_seed_value = st.number_input(
                "Semilla ADN", value=7, step=1, disabled=not dna_seed_option, key="dna_seed_value"
            )
            random_length = st.slider(
                "Longitud de secuencia aleatoria",
                min_value=30,
                max_value=300,
                value=len(default_dna_sequence),
                step=5,
                key="dna_random_length",
            )
            if st.button("Generar secuencia aleatoria", key="dna_random_button"):
                rng = np.random.default_rng(int(dna_seed_value) if dna_seed_option else None)
                st.session_state["dna_sequence"] = "".join(rng.choice(DNA_BASES, size=int(random_length)))

            dna_sequence_input = st.text_area(
                "Secuencia inicial (A/C/G/T)", st.session_state["dna_sequence"], height=140, key="dna_sequence_area"
            )
            clean_sequence = "".join([base for base in dna_sequence_input.strip().upper() if base in DNA_BASES])
            if not clean_sequence:
                clean_sequence = default_dna_sequence
            st.session_state["dna_sequence"] = clean_sequence

    image_tab, dna_tab = st.tabs(["Imagen / memoria", "ADN biosimulation"])

    with image_tab:
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
            "---------------------------------",
            options=frame_labels,
            value=frame_labels[-1],
        )
        selected_index = frame_labels.index(selected_label)
        current_frame = frames[selected_index]
        latest_frame = frames[-1]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Imagen original")
            st.image(image, use_container_width=True)
        with col2:
            st.subheader("Imagen corrompida / animación")
            anim_placeholder = st.empty()
            anim_placeholder.image(
                current_frame.image, use_container_width=True, caption=f"t = {current_frame.time_hours:.2f} h"
            )

            if st.button("Reproducir animación", type="primary"):
                for frame in frames:
                    anim_placeholder.image(
                        frame.image,
                        use_container_width=True,
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
""        )



    with dna_tab:
        st.subheader("Biosimulación: mutaciones de ADN por rayos cósmicos")
        st.write(
            "Modelo alternativo donde cada base del ADN se comporta como un bit susceptible a mutar; la tasa escala con la altitud "
            "como $2^{h/1000}$ y las mutaciones se generan con un proceso de Poisson."
        )

        dna_sequence = st.session_state["dna_sequence"]
        dna_seed = int(dna_seed_value) if dna_seed_option else None

        dna_frames = simulate_dna_mutations_over_time(
            sequence=dna_sequence,
            altitude_m=float(dna_altitude),
            total_hours=float(dna_hours),
            steps=int(dna_steps),
            base_rate_per_base=float(dna_base_rate),
            seed=dna_seed,
        )

        dna_labels = [f"{frame.time_hours:.2f} h" for frame in dna_frames]
        dna_selected = st.select_slider(
            "Explora las mutaciones a lo largo del tiempo", options=dna_labels, value=dna_labels[-1]
        )
        dna_index = dna_labels.index(dna_selected)
        dna_frame = dna_frames[dna_index]
        dna_latest = dna_frames[-1]

        col_a, col_b = st.columns(2)
        with col_a:
            st.caption("Secuencia inicial")
            st.code("\n".join(_wrap_sequence(dna_sequence)), language="text")
        with col_b:
            st.caption("Secuencia mutada / animación")
            st.markdown(
                _highlight_sequence(dna_frame.sequence, dna_sequence),
                unsafe_allow_html=True,
            )
            st.caption("Bases mutadas resaltadas en amarillo")

        st.metric("Longitud (bases)", f"{len(dna_sequence):,}")
        st.metric("Mutaciones esperadas (μ)", f"{dna_latest.expected_mutations:.3e}")
        st.metric("Mutaciones observadas", f"{dna_latest.observed_mutations}")
        st.metric("Prob. ≥ 1 mutación", f"{dna_latest.probability_at_least_one:.2%}")

        st.subheader("Probabilidad acumulada de mutación vs tiempo")
        dna_hours_range = np.linspace(0, max(dna_hours, 24.0), 40)
        dna_prob_curve = mutation_probability_curve(
            base_rate_per_base=float(dna_base_rate),
            altitude_m=float(dna_altitude),
            num_bases=len(dna_sequence),
            hours_points=dna_hours_range,
        )
        dna_df = pd.DataFrame({"Horas": dna_hours_range, "Probabilidad": dna_prob_curve})
        st.line_chart(dna_df, x="Horas", y="Probabilidad")




if __name__ == "__main__":
    main()
