from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from app_utils import (
    describe_lab9_image,
    lab_summary_cards,
    load_fashion_items,
    load_lab9_images,
    search_fashion_items,
)


ROOT = Path(__file__).parent
LAB6_IMAGE_DIR = ROOT / "lab6_clip_multimodal_fashion_search" / "fashion_images"
LAB9_DIR = ROOT / "lab9_image_captioning"


st.set_page_config(
    page_title="Transformer Practice Labs",
    layout="wide",
)


st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .small-note { color: #5f6368; font-size: 0.92rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def render_header() -> None:
    st.title("Transformer Practice Labs")
    st.caption("Lab 6 fashion search and Lab 9 image captioning demo for Streamlit deployment.")

    cards = lab_summary_cards()
    cols = st.columns(len(cards))
    for col, (lab, title, body) in zip(cols, cards):
        with col:
            st.metric(lab, title)
            st.markdown(f"<p class='small-note'>{body}</p>", unsafe_allow_html=True)


def render_fashion_grid(items, columns: int = 4) -> None:
    if not items:
        st.warning("No fashion images were found.")
        return

    cols = st.columns(columns)
    for index, item in enumerate(items):
        with cols[index % columns]:
            st.image(str(item.path), caption=item.label, use_container_width=True)


def render_lab6() -> None:
    st.subheader("Lab 6: CLIP Fashion Search")
    st.write(
        "The notebook uses CLIP embeddings for image-text search. "
        "This deployment keeps the interaction light so the app opens reliably on Streamlit Cloud."
    )

    items = load_fashion_items(LAB6_IMAGE_DIR)
    left, right = st.columns([1, 2])

    with left:
        query = st.text_input("Style query", value="화이트 스니커즈")
        limit = st.slider("Results", min_value=3, max_value=12, value=6)
        selected_label = st.selectbox(
            "Image-to-image seed",
            options=[item.label for item in items],
            index=0 if items else None,
        )

        st.markdown("**Model idea**")
        st.write(
            "Text and images are converted into vectors. Search ranks images by vector similarity."
        )

    with right:
        st.markdown("**Text search preview**")
        results = search_fashion_items(query, items, limit=limit)
        render_fashion_grid(results, columns=3)

    if selected_label:
        selected = next(item for item in items if item.label == selected_label)
        st.divider()
        st.markdown("**Image-to-image preview**")
        seed_col, results_col = st.columns([1, 3])
        with seed_col:
            st.image(str(selected.path), caption=f"Seed: {selected.label}", use_container_width=True)
        with results_col:
            related = search_fashion_items(selected.label, items, limit=4)
            render_fashion_grid(related, columns=4)


def render_lab9() -> None:
    st.subheader("Lab 9: Image Captioning")
    st.write(
        "The notebook uses BLIP for caption generation, BLEU for evaluation, and a translation model "
        "for Korean output. This app shows the deployment-friendly demo flow with the local assets."
    )

    images = load_lab9_images(LAB9_DIR)
    if not images:
        st.warning("No Lab 9 images were found.")
        return

    labels = [image.label for image in images]
    selected_label = st.selectbox("Caption image", options=labels)
    selected = next(image for image in images if image.label == selected_label)
    caption = describe_lab9_image(selected)

    image_col, caption_col = st.columns([1, 1])
    with image_col:
        st.image(str(selected.path), caption=selected.path.name, use_container_width=True)

    with caption_col:
        st.markdown("**Caption preview**")
        st.write(caption.english)
        st.markdown("**Korean preview**")
        st.write(caption.korean)

        st.markdown("**Notebook pipeline**")
        pipeline = pd.DataFrame(
            [
                {"Step": "Input", "Role": "Load image"},
                {"Step": "BLIP", "Role": "Generate English caption"},
                {"Step": "BLEU", "Role": "Compare generated caption with references"},
                {"Step": "MarianMT", "Role": "Translate caption into Korean"},
            ]
        )
        st.dataframe(pipeline, hide_index=True, use_container_width=True)

    st.divider()
    st.markdown("**Sample image set**")
    render_fashion_grid(images[:8], columns=4)


def render_deployment_notes() -> None:
    with st.sidebar:
        st.header("Deployment")
        st.write("Repository")
        st.code("pastar20250101-creator/practice_transformer4")
        st.write("Entrypoint")
        st.code("streamlit_app.py")
        st.write("Python dependencies")
        st.code("requirements.txt")
        st.divider()
        st.write("Lab 6 and Lab 9 notebooks remain in the repository for the full model code.")


def main() -> None:
    render_header()
    render_deployment_notes()

    tab_lab6, tab_lab9 = st.tabs(["Lab 6 Fashion Search", "Lab 9 Image Captioning"])
    with tab_lab6:
        render_lab6()
    with tab_lab9:
        render_lab9()


if __name__ == "__main__":
    main()
