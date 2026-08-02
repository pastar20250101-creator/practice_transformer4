from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from app_utils import (
    LAB1_DIR,
    analyze_movie_reviews,
    caption_uploaded_image,
    fashion_profiles,
    load_caption_records,
    load_token_records,
    load_translation_pairs,
    my_photo_caption_record,
    movie_review_sample_rows,
    movie_review_samples,
    my_photo_path,
    search_fashion_profiles,
    translate_by_similarity,
)


ROOT = Path(__file__).parent


st.set_page_config(
    page_title="AI Lab Tools",
    page_icon="AI",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f6f7f4;
        color: #17252a;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1.1rem;
        padding-bottom: 2.2rem;
    }
    h1, h2, h3 {
        letter-spacing: 0;
    }
    .topbar {
        align-items: center;
        background: #15191f;
        border: 1px solid #26313a;
        border-radius: 8px;
        color: #ffffff;
        display: flex;
        gap: 1rem;
        justify-content: space-between;
        margin-bottom: 1rem;
        padding: 1rem 1.15rem;
    }
    .topbar h1 {
        color: #ffffff;
        font-size: 1.55rem;
        line-height: 1.2;
        margin: 0;
    }
    .topbar span {
        color: #8fd6cc;
        font-size: 0.86rem;
        font-weight: 800;
        white-space: nowrap;
    }
    .tool-card {
        background: #ffffff;
        border: 1px solid #dde5e5;
        border-radius: 8px;
        min-height: 108px;
        padding: 0.9rem;
    }
    .tool-card strong {
        color: #111820;
        display: block;
        font-size: 0.98rem;
        margin-bottom: 0.35rem;
    }
    .tool-card p {
        color: #53636b;
        font-size: 0.84rem;
        line-height: 1.45;
        margin: 0;
    }
    .result-box {
        background: #ffffff;
        border: 1px solid #dde5e5;
        border-radius: 8px;
        padding: 0.85rem;
    }
    .result-box strong {
        color: #8f1d22;
        display: block;
        font-size: 0.78rem;
        letter-spacing: 0.02em;
        margin-bottom: 0.35rem;
        text-transform: uppercase;
    }
    .result-box p {
        color: #344047;
        line-height: 1.5;
        margin: 0.25rem 0;
    }
    .image-card {
        background: #ffffff;
        border: 1px solid #dde5e5;
        border-radius: 8px;
        margin-bottom: 0.85rem;
        overflow: hidden;
    }
    .image-card-body {
        padding: 0.7rem;
    }
    .image-card strong {
        color: #17252a;
        display: block;
        font-size: 0.95rem;
        margin-bottom: 0.3rem;
    }
    .image-card p {
        color: #52616b;
        font-size: 0.82rem;
        line-height: 1.42;
        margin: 0.2rem 0;
    }
    .badge {
        background: #edf5f3;
        border-radius: 999px;
        color: #1f6f65;
        display: inline-block;
        font-size: 0.74rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
        padding: 0.18rem 0.5rem;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.45rem;
    }
    @media (max-width: 760px) {
        .topbar {
            align-items: flex-start;
            flex-direction: column;
        }
        .topbar span {
            white-space: normal;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def asset_path(path: Path) -> str:
    return str(ROOT / path)


@st.cache_data(show_spinner=False)
def cached_token_frame(limit: int = 20) -> pd.DataFrame:
    records = load_token_records(ROOT / LAB1_DIR / "output" / "improved_top_tokens.txt", limit=limit)
    return pd.DataFrame(
        [
            {
                "순위": token.rank,
                "키워드": token.word,
                "빈도": token.count,
                "비율": token.percent,
            }
            for token in records
        ]
    )


@st.cache_data(show_spinner=False)
def cached_translation_pairs(limit: int = 300):
    return load_translation_pairs(limit=limit)


@st.cache_data(show_spinner=False)
def cached_caption_records():
    return list(load_caption_records())


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def render_header() -> None:
    st.markdown(
        """
        <section class="topbar">
            <h1>AI Lab Tools</h1>
            <span>Lab 1 · Lab 2 · Lab 6 · Lab 9</span>
        </section>
        """,
        unsafe_allow_html=True,
    )

    cards = st.columns(4)
    card_texts = [
        ("영화 리뷰 분석기", "리뷰를 입력하면 감성 점수와 핵심 키워드를 바로 확인한다."),
        ("번역기", "학습 문장과 가장 가까운 문장을 찾아 영어 번역을 보여준다."),
        ("패션 검색기", "검색어와 카테고리로 의상 이미지를 빠르게 찾는다."),
        ("이미지 캡셔닝", "COCO 샘플과 사용자 이미지를 캡션 화면으로 확인한다."),
    ]
    for column, (title, body) in zip(cards, card_texts):
        with column:
            st.markdown(
                f"""
                <div class="tool-card">
                    <strong>{escape(title)}</strong>
                    <p>{escape(body)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_lab1() -> None:
    st.subheader("영화 리뷰 분석기")

    samples = movie_review_samples()
    sample_rows = movie_review_sample_rows(samples)
    movie_names = ["직접 입력"] + [sample.movie for sample in samples]

    control_col, result_col = st.columns([0.85, 1.15])
    with control_col:
        selected_movie = st.selectbox("영화 선택", movie_names)
        default_review = samples[0].review
        if selected_movie != "직접 입력":
            default_review = next(sample.review for sample in samples if sample.movie == selected_movie)
        review_text = st.text_area("리뷰 입력", value=default_review, height=160)

    result = analyze_movie_reviews(review_text)
    keyword_counts = result["keyword_counts"]
    keyword_frame = pd.DataFrame(
        [{"키워드": word, "빈도": count} for word, count in keyword_counts.most_common(10)]
    )

    with result_col:
        metrics = st.columns(5)
        metrics[0].metric("감성", str(result["label"]))
        metrics[1].metric("점수", int(result["score"]))
        metrics[2].metric("긍정", int(result["positive_hits"]))
        metrics[3].metric("부정", int(result["negative_hits"]))
        metrics[4].metric("토큰", int(result["token_count"]))

        chart_col, table_col = st.columns([1.2, 0.8])
        with chart_col:
            st.markdown("#### 입력 리뷰 키워드")
            if keyword_frame.empty:
                st.empty()
            else:
                st.bar_chart(keyword_frame.set_index("키워드")["빈도"])
        with table_col:
            st.markdown("#### TOP 10")
            st.dataframe(keyword_frame, hide_index=True, width="stretch", height=260)

    st.markdown("#### 영화별 샘플 분석")
    st.dataframe(pd.DataFrame(sample_rows), hide_index=True, width="stretch", height=245)

    st.markdown("#### 학습 코퍼스 키워드")
    corpus_frame = cached_token_frame(limit=20)
    st.bar_chart(corpus_frame.set_index("키워드")["빈도"])


def render_lab2() -> None:
    st.subheader("한국어 → 영어 번역기")

    pairs = cached_translation_pairs(limit=300)
    default_text = pairs[0].korean if pairs else ""

    input_col, result_col = st.columns([0.85, 1.15])
    with input_col:
        query = st.text_area("한국어 입력", value=default_text, height=150)
        sample_count = st.slider("검색 문장 수", 50, 300, 300, step=50)

    active_pairs = pairs[:sample_count]
    result = translate_by_similarity(query, active_pairs)

    with result_col:
        metrics = st.columns(3)
        metrics[0].metric("데이터", len(active_pairs))
        metrics[1].metric("유사도", f"{float(result['similarity']) * 100:.1f}%")
        metrics[2].metric("방식", "검색형")

        left, right = st.columns(2)
        with left:
            st.markdown("#### 번역 결과")
            st.markdown(
                f"<div class='result-box'><strong>English</strong><p>{escape(result['translation'])}</p></div>",
                unsafe_allow_html=True,
            )
        with right:
            st.markdown("#### 매칭 문장")
            st.markdown(
                f"<div class='result-box'><strong>Korean</strong><p>{escape(result['source'])}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### 예시 문장")
    rows = [{"한국어": pair.korean, "영어": pair.english} for pair in active_pairs[:10]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", height=310)


def render_lab6() -> None:
    st.subheader("패션 검색기")

    profiles = fashion_profiles()
    categories = sorted({profile.category for profile in profiles})

    control_col, result_col = st.columns([0.75, 2.25])
    with control_col:
        query = st.text_input("검색어", value="minimal office")
        selected_categories = st.multiselect("카테고리", categories, default=categories)
        limit = st.slider("결과 수", 3, 12, 9)

    filtered_profiles = [profile for profile in profiles if profile.category in selected_categories]
    results = search_fashion_profiles(query, filtered_profiles, limit=limit)

    with result_col:
        metrics = st.columns(3)
        metrics[0].metric("이미지", len(filtered_profiles))
        metrics[1].metric("결과", len(results))
        metrics[2].metric("카테고리", len(selected_categories))

        if not results:
            st.markdown("<div class='result-box'><p>검색 결과가 없습니다.</p></div>", unsafe_allow_html=True)
            return

        grid = st.columns(3)
        for index, profile in enumerate(results):
            with grid[index % 3]:
                st.image(asset_path(profile.image_path), width="stretch")
                st.markdown(
                    f"""
                    <div class="image-card">
                        <div class="image-card-body">
                            <span class="badge">{escape(profile.category)}</span>
                            <strong>{escape(profile.name)}</strong>
                            <p>{escape(profile.description)}</p>
                            <p>{escape(", ".join(profile.tags[:4]))}</p>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_lab9() -> None:
    st.subheader("이미지 캡셔닝")

    records = cached_caption_records()
    user_photo = my_photo_path()
    names = [record.image_name for record in records]
    if (ROOT / user_photo).exists():
        names.insert(0, "my_photo.png")

    control_col, result_col = st.columns([0.75, 2.25])
    with control_col:
        uploaded = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])
        selected = st.selectbox("샘플 이미지", names)
        max_caption = st.slider("캡션 수", 1, 5, 3)

    with result_col:
        mode_label = "Upload" if uploaded is not None else "Scene" if selected == "my_photo.png" else "COCO"
        metrics = st.columns(4)
        metrics[0].metric("샘플", len(records))
        metrics[1].metric("영어 캡션", max_caption)
        metrics[2].metric("한국어 캡션", max_caption)
        metrics[3].metric("모드", mode_label)

        image_col, text_col = st.columns([1, 1.35])
        with image_col:
            if uploaded is not None:
                uploaded_bytes = uploaded.getvalue()
                st.image(uploaded_bytes, width="stretch")
                preview = caption_uploaded_image(uploaded_bytes)
                english = (preview.english,)
                korean = (preview.korean,)
            elif selected == "my_photo.png":
                record = my_photo_caption_record()
                st.image(asset_path(record.image_path), width="stretch")
                english = record.english_captions[:max_caption]
                korean = record.korean_captions[:max_caption]
            else:
                record = next(item for item in records if item.image_name == selected)
                st.image(asset_path(record.image_path), width="stretch")
                english = record.english_captions[:max_caption]
                korean = record.korean_captions[:max_caption]

        with text_col:
            st.markdown("#### English")
            st.markdown(
                "<div class='result-box'><strong>Caption</strong>"
                + "".join(f"<p>{escape(sentence)}</p>" for sentence in english)
                + "</div>",
                unsafe_allow_html=True,
            )
            st.markdown("#### Korean")
            st.markdown(
                "<div class='result-box'><strong>Caption</strong>"
                + "".join(f"<p>{escape(sentence)}</p>" for sentence in korean)
                + "</div>",
                unsafe_allow_html=True,
            )

    st.markdown("#### 이미지 갤러리")
    grid = st.columns(5)
    for index, record in enumerate(records[:10]):
        with grid[index % 5]:
            st.image(asset_path(record.image_path), width="stretch")


def main() -> None:
    render_header()

    tab1, tab2, tab6, tab9 = st.tabs(
        ["Lab 1 · 영화 리뷰 분석기", "Lab 2 · 번역기", "Lab 6 · 패션 검색기", "Lab 9 · 이미지 캡셔닝"]
    )
    with tab1:
        render_lab1()
    with tab2:
        render_lab2()
    with tab6:
        render_lab6()
    with tab9:
        render_lab9()


if __name__ == "__main__":
    main()
