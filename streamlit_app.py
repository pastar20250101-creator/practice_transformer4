from __future__ import annotations

import base64
from collections import Counter
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

from app_utils import (
    FashionLook,
    describe_lab9_image,
    filter_reviews,
    get_fashion_looks,
    keyword_counts,
    lab_summary_cards,
    load_movie_reviews,
    review_summary,
    search_fashion_looks,
)


ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
MOVIE_REVIEW_CSV = DATA_DIR / "movie_reviews.csv"
LAB9_DIR = ROOT / "lab9_image_captioning"
NSMC_URL = "https://github.com/e9t/nsmc"
MENDELEY_URL = "https://data.mendeley.com/datasets/jb5knzh8yv/6"
UNSPLASH_URL = "https://unsplash.com/"


st.set_page_config(
    page_title="AI Lab Showcase",
    page_icon="AI",
    layout="wide",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #f5f7f6;
        color: #17252a;
    }
    .block-container {
        max-width: 1240px;
        padding-top: 1.1rem;
        padding-bottom: 2.5rem;
    }
    h1, h2, h3, h4 {
        letter-spacing: 0;
    }
    section[data-testid="stSidebar"] {
        background: #11161c;
    }
    section[data-testid="stSidebar"] * {
        color: #f8fafc;
    }
    section[data-testid="stSidebar"] code {
        color: #d9e7e4;
    }
    .hero-shell {
        background: #11161c;
        border: 1px solid #202a35;
        border-radius: 8px;
        color: #ffffff;
        display: grid;
        gap: 1.2rem;
        grid-template-columns: minmax(0, 1.45fr) minmax(280px, 0.75fr);
        margin-bottom: 1.25rem;
        padding: 1.5rem;
    }
    .hero-shell h1 {
        color: #ffffff;
        font-size: 2.45rem;
        line-height: 1.05;
        margin: 0.25rem 0 0.75rem;
    }
    .hero-copy {
        color: #d7dee3;
        font-size: 1rem;
        line-height: 1.65;
        margin: 0;
        max-width: 760px;
    }
    .hero-panel {
        background: #f7faf8;
        border: 1px solid #dce6eb;
        border-radius: 8px;
        color: #17252a;
        padding: 1rem;
    }
    .hero-panel b {
        display: block;
        font-size: 0.82rem;
        margin-bottom: 0.55rem;
        text-transform: uppercase;
    }
    .status-row {
        align-items: center;
        border-top: 1px solid #dce6eb;
        display: flex;
        justify-content: space-between;
        padding: 0.55rem 0;
    }
    .status-row:first-of-type {
        border-top: 0;
    }
    .status-row span {
        color: #52616b;
        font-size: 0.86rem;
    }
    .status-row strong {
        color: #0f766e;
        font-size: 0.92rem;
    }
    .kicker {
        color: #44d2bd;
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
    }
    .lead {
        color: #344047;
        font-size: 1.02rem;
        line-height: 1.65;
        margin-bottom: 1rem;
    }
    .lab-card, .info-tile, .fashion-card, .signal-card, .caption-box {
        border: 1px solid #dce6eb;
        border-radius: 8px;
        background: #ffffff;
    }
    .lab-card {
        border-top: 4px solid #0f766e;
        padding: 1rem;
        min-height: 190px;
    }
    .lab-card span {
        color: #9b2226;
        display: block;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }
    .lab-card strong {
        color: #17252a;
        display: block;
        font-size: 1.05rem;
        margin-bottom: 0.35rem;
    }
    .lab-card p {
        color: #4f5b62;
        font-size: 0.88rem;
        line-height: 1.45;
        margin: 0.25rem 0;
    }
    .info-tile {
        padding: 0.85rem 0.95rem;
        min-height: 112px;
    }
    .signal-card {
        min-height: 132px;
        padding: 0.95rem;
    }
    .signal-card span {
        color: #52616b;
        display: block;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .signal-card strong {
        color: #17252a;
        display: block;
        font-size: 1.35rem;
        margin: 0.25rem 0;
    }
    .signal-card p {
        color: #5d6b72;
        font-size: 0.86rem;
        line-height: 1.45;
        margin: 0;
    }
    .info-tile b {
        color: #17252a;
    }
    .info-tile p {
        color: #4f5b62;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0.35rem 0 0;
    }
    .fashion-card {
        overflow: hidden;
        padding: 0;
        min-height: 365px;
    }
    .fashion-image {
        aspect-ratio: 4 / 5;
        background: #dce6eb;
        display: block;
        object-fit: cover;
        width: 100%;
    }
    .fashion-body {
        padding: 0.8rem;
    }
    .fashion-title {
        color: #17252a;
        font-size: 0.98rem;
        font-weight: 800;
        margin-bottom: 0.25rem;
    }
    .fashion-meta {
        color: #5d6b72;
        font-size: 0.84rem;
        line-height: 1.45;
        min-height: 68px;
    }
    .tag-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.28rem;
        margin-top: 0.55rem;
    }
    .tag-chip {
        background: #eef4f3;
        border: 1px solid #d6e4e1;
        border-radius: 999px;
        color: #315d57;
        font-size: 0.72rem;
        padding: 0.12rem 0.45rem;
    }
    .source-note {
        color: #5d6b72;
        font-size: 0.86rem;
        line-height: 1.45;
    }
    .caption-box {
        padding: 1rem;
    }
    .caption-box span {
        color: #9b2226;
        display: block;
        font-size: 0.8rem;
        font-weight: 800;
        margin-bottom: 0.3rem;
        text-transform: uppercase;
    }
    .caption-box p {
        color: #344047;
        line-height: 1.55;
        margin: 0;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }
    [data-testid="stMetricLabel"] {
        color: #52616b;
    }
    @media (max-width: 760px) {
        .hero-shell {
            grid-template-columns: 1fr;
            padding: 1rem;
        }
        .hero-shell h1 {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def counter_to_frame(counter: Counter[str], key_name: str, value_name: str = "count") -> pd.DataFrame:
    return pd.DataFrame(
        [{key_name: key, value_name: value} for key, value in counter.most_common()]
    )


@st.cache_data(show_spinner=False)
def local_image_data_uri(path_text: str) -> str:
    path = Path(path_text)
    if not path.is_absolute():
        path = ROOT / path

    mime_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def image_source(image: str) -> str:
    if image.startswith("http://") or image.startswith("https://"):
        return image

    path = Path(image)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return ""
    return local_image_data_uri(str(path))


def render_sidebar() -> None:
    with st.sidebar:
        st.header("AI Lab Showcase")
        st.write("포함 실습")
        st.code("Lab 1\nLab 6\nLab 9", language="text")
        st.divider()
        st.header("Streamlit 배포")
        st.code(
            "Repository: pastar20250101-creator/practice_transformer4\n"
            "Branch: main\n"
            "Main file path: streamlit_app.py",
            language="text",
        )
        st.divider()
        st.header("출처")
        st.markdown(f"- [NSMC]({NSMC_URL})")
        st.markdown(f"- [Mendeley Korean Movie Reviews]({MENDELEY_URL})")
        st.markdown(f"- [Unsplash]({UNSPLASH_URL})")
        st.caption("무단 대량 크롤링 없이 로컬 CSV와 공개 이미지 URL을 사용합니다.")


def render_overview() -> None:
    cards = lab_summary_cards()
    reviews = load_movie_reviews(MOVIE_REVIEW_CSV)
    looks = get_fashion_looks()
    total_summary = review_summary(reviews, "전체")
    top_keywords = ", ".join(keyword for keyword, _ in keyword_counts(reviews).most_common(3))

    st.markdown(
        f"""
        <section class="hero-shell">
            <div>
                <div class="kicker">AI Human Transformer Practice</div>
                <h1>AI Lab Showcase</h1>
                <p class="hero-copy">
                    과제 1, 6, 9만 제출용으로 정리한 Streamlit 대시보드입니다.
                    영화 리뷰의 텍스트 분석, 실제 의상 이미지 기반 검색, 이미지 캡셔닝 파이프라인을
                    무거운 모델 다운로드 없이 빠르게 시연할 수 있게 구성했습니다.
                </p>
            </div>
            <div class="hero-panel">
                <b>Submission Focus</b>
                <div class="status-row"><span>Included labs</span><strong>1, 6, 9 only</strong></div>
                <div class="status-row"><span>Cloud startup</span><strong>Lightweight</strong></div>
                <div class="status-row"><span>Secrets policy</span><strong>No keys in code</strong></div>
                <div class="status-row"><span>Model runtime</span><strong>Notebook only</strong></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    signal_cols = st.columns(4)
    signals = [
        ("Labs", "3", "Lab 1, Lab 6, Lab 9만 앱에 표시"),
        ("Movie reviews", str(len(reviews)), f"평균 평점 {total_summary['average_rating']}점"),
        ("Fashion looks", str(len(looks)), "로컬 이미지와 공개 이미지 URL 결합"),
        ("Top keywords", top_keywords or "-", "전처리와 토큰화 결과 미리보기"),
    ]
    for col, (label, value, body) in zip(signal_cols, signals):
        with col:
            st.markdown(
                f"""
                <div class="signal-card">
                    <span>{label}</span>
                    <strong>{value}</strong>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("#### 실습 구성")
    cols = st.columns(3)
    for col, card in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="lab-card">
                    <span>{card.lab_id}</span>
                    <strong>{card.title}</strong>
                    <p>{card.subtitle}</p>
                    <p><b>학습:</b> {card.study_focus}</p>
                    <p><b>결과:</b> {card.artifact}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_info_tiles(items: list[tuple[str, str]]) -> None:
    cols = st.columns(len(items))
    for col, (title, body) in zip(cols, items):
        with col:
            st.markdown(
                f"<div class='info-tile'><b>{title}</b><p>{body}</p></div>",
                unsafe_allow_html=True,
            )


def render_movie_lab() -> None:
    reviews = load_movie_reviews(MOVIE_REVIEW_CSV)
    movie_titles = ["전체"] + sorted({review.movie_title for review in reviews})

    st.subheader("Lab 1: 실제 영화 리뷰 분석")
    st.write(
        "실제 영화명을 기준으로 구성한 리뷰 CSV를 읽고, 영화별 감성 비율과 키워드를 분석합니다. "
        "웹사이트를 실시간 크롤링하지 않고 로컬 스냅샷을 사용해 배포 안정성을 확보했습니다."
    )

    render_info_tiles(
        [
            ("전처리", "문장 부호를 제거하고 분석 가능한 단어만 남깁니다."),
            ("토큰화", "리뷰 문장을 단어 단위로 나누어 빈도 분석에 사용합니다."),
            ("감성 분석", "평점 기반 positive/negative 라벨로 긍정/부정 비율을 계산합니다."),
        ]
    )

    movie_overview = pd.DataFrame(
        [
            {
                "영화": title,
                "리뷰 수": review_summary(reviews, title)["review_count"],
                "평균 평점": review_summary(reviews, title)["average_rating"],
                "긍정 비율": review_summary(reviews, title)["positive_ratio"],
                "부정 비율": review_summary(reviews, title)["negative_ratio"],
            }
            for title in sorted({review.movie_title for review in reviews})
        ]
    )

    st.markdown("##### 영화별 요약")
    overview_col, ratio_col = st.columns([1.1, 1.4])
    with overview_col:
        st.dataframe(movie_overview, hide_index=True, use_container_width=True)
    with ratio_col:
        st.bar_chart(movie_overview.set_index("영화")[["긍정 비율", "부정 비율"]])

    st.divider()
    control_col, chart_col = st.columns([0.8, 2.2])
    with control_col:
        selected_movie = st.selectbox("영화 선택", movie_titles)
        selected_reviews = filter_reviews(reviews, selected_movie)
        summary = review_summary(reviews, selected_movie)
        st.metric("리뷰 수", summary["review_count"])
        st.metric("평균 평점", summary["average_rating"])
        st.metric("긍정 비율", f"{summary['positive_ratio'] * 100:.1f}%")
        st.metric("부정 비율", f"{summary['negative_ratio'] * 100:.1f}%")
        st.caption("평점 8점 이상은 긍정, 4점 이하는 부정으로 정리한 제출용 샘플입니다.")

    with chart_col:
        st.markdown("##### 키워드 TOP 10")
        keyword_frame = counter_to_frame(keyword_counts(selected_reviews), "keyword").head(10)
        st.bar_chart(keyword_frame.set_index("keyword"))

        sentiment_frame = pd.DataFrame(
            [
                {"sentiment": "positive", "ratio": summary["positive_ratio"]},
                {"sentiment": "negative", "ratio": summary["negative_ratio"]},
            ]
        )
        st.markdown("##### 감성 비율")
        st.bar_chart(sentiment_frame.set_index("sentiment"))

    st.markdown("##### 긍정 / 부정 키워드 비교")
    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.dataframe(
            counter_to_frame(keyword_counts(selected_reviews, sentiment="positive"), "positive keyword").head(10),
            hide_index=True,
            use_container_width=True,
        )
    with neg_col:
        st.dataframe(
            counter_to_frame(keyword_counts(selected_reviews, sentiment="negative"), "negative keyword").head(10),
            hide_index=True,
            use_container_width=True,
        )

    st.markdown("##### 리뷰 원문")
    review_frame = pd.DataFrame(
        [
            {
                "영화": review.movie_title,
                "연도": review.release_year,
                "평점": review.rating,
                "감성": review.sentiment,
                "리뷰": review.review_text,
                "출처": review.source,
            }
            for review in selected_reviews
        ]
    )
    st.dataframe(review_frame, hide_index=True, use_container_width=True)

    st.info(
        "출처 메모: NSMC와 Mendeley Korean Movie Reviews 같은 공개 영화 리뷰 데이터셋 구조를 참고한 로컬 CSV 스냅샷입니다. "
        "서비스 약관 이슈를 피하기 위해 앱 실행 중 실시간 무단 크롤링은 하지 않습니다."
    )


def render_fashion_card(look: FashionLook) -> None:
    src = image_source(look.image)
    image_html = (
        f'<img class="fashion-image" src="{escape(src)}" loading="lazy" alt="{escape(look.title)}">'
        if src
        else '<div class="fashion-image"></div>'
    )
    tags = "".join(
        f'<span class="tag-chip">#{escape(tag.replace(" ", "_"))}</span>'
        for tag in look.tags[:4]
    )
    source = "Unsplash" if look.image.startswith("https://") else "Local lab asset"

    st.markdown(
        f"""
        <article class="fashion-card">
            {image_html}
            <div class="fashion-body">
                <div class="fashion-title">{escape(look.title)}</div>
                <div class="fashion-meta">
                    <b>{escape(look.category)}</b> | {escape(look.palette)}<br>
                    {escape(look.styling_note)}<br>
                    <span class="source-note">Source: {escape(source)}</span>
                </div>
                <div class="tag-row">{tags}</div>
            </div>
        </article>
        """,
        unsafe_allow_html=True,
    )


def render_fashion_grid(looks: list[FashionLook], columns: int = 4) -> None:
    cols = st.columns(columns)
    for index, look in enumerate(looks):
        with cols[index % columns]:
            render_fashion_card(look)


def render_fashion_lab() -> None:
    looks = get_fashion_looks()
    categories = sorted({look.category for look in looks})

    st.subheader("Lab 6: 실제 의상 이미지 기반 패션 검색")
    st.write(
        "로컬 과제 이미지에 공개 패션 이미지 URL을 더해 20개 이상의 스타일 보드를 구성했습니다. "
        "Streamlit에서는 태그 기반 검색으로 CLIP의 텍스트-이미지 검색 흐름을 가볍게 시연합니다."
    )

    render_info_tiles(
        [
            ("이미지 임베딩", "실제 CLIP은 이미지를 숫자 벡터로 변환합니다."),
            ("텍스트 임베딩", "검색 문장도 같은 공간의 벡터로 변환합니다."),
            ("Cosine similarity", "두 벡터가 얼마나 같은 방향인지 계산해 검색 순위를 만듭니다."),
        ]
    )

    collection_cols = st.columns(3)
    collection_signals = [
        ("Total looks", str(len(looks)), "20개 이상의 패션 이미지 메타데이터"),
        ("Categories", str(len(categories)), "streetwear부터 accessories까지"),
        (
            "Search mode",
            "Tag ranking",
            "Cloud 배포에서는 CLIP 대신 가벼운 태그 유사도 사용",
        ),
    ]
    for col, (label, value, body) in zip(collection_cols, collection_signals):
        with col:
            st.markdown(
                f"""
                <div class="signal-card">
                    <span>{label}</span>
                    <strong>{value}</strong>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()
    control_col, result_col = st.columns([0.9, 2.1])
    with control_col:
        query = st.text_input("검색어", value="black street sneaker")
        category = st.selectbox("카테고리", ["All"] + categories)
        limit = st.slider("결과 개수", 4, 20, 8)
        st.caption("예시: black street sneaker, minimal office, dress, bag, trench coat")
        st.markdown(
            "<p class='source-note'>카테고리: streetwear, minimal, office, casual, dress, sneakers, outer, accessories. "
            "원격 이미지는 공개 이미지 URL이며 출처를 카드에 보존합니다.</p>",
            unsafe_allow_html=True,
        )

    filtered = looks if category == "All" else [look for look in looks if look.category == category]
    results = search_fashion_looks(query, filtered, limit=limit)

    with result_col:
        st.markdown("##### 검색 결과")
        render_fashion_grid(results, columns=4)

    st.markdown("##### 카테고리 분포")
    category_frame = pd.DataFrame(
        Counter(look.category for look in looks).most_common(),
        columns=["category", "count"],
    )
    st.bar_chart(category_frame.set_index("category"))


def lab9_images() -> list[Path]:
    images: list[Path] = []
    user_photo = LAB9_DIR / "my_photo.png"
    if user_photo.exists():
        images.append(user_photo)
    images.extend(sorted((LAB9_DIR / "coco_images").glob("*.jpg"))[:24])
    return images


def render_caption_lab() -> None:
    images = lab9_images()

    st.subheader("Lab 9: 이미지 캡셔닝 데모")
    st.write(
        "이미지를 선택하면 영어 캡션 preview와 한국어 설명 preview를 보여줍니다. "
        "실제 노트북에서는 BLIP, BLEU, 번역 모델 흐름을 실행합니다."
    )

    render_info_tiles(
        [
            ("Vision encoder", "이미지의 객체와 장면 정보를 벡터 표현으로 바꿉니다."),
            ("Language decoder", "이미지 표현을 바탕으로 단어를 순서대로 생성합니다."),
            ("BLEU 평가", "생성 문장과 기준 문장의 겹침을 확인합니다."),
        ]
    )

    st.divider()
    if not images:
        st.warning("Lab 9 이미지가 없습니다.")
        return

    selected_name = st.selectbox("이미지 선택", [image.name for image in images])
    selected_path = next(path for path in images if path.name == selected_name)
    caption = describe_lab9_image(selected_name)

    preview_col, text_col = st.columns([1, 1.1])
    with preview_col:
        st.image(str(selected_path), caption=selected_path.name, use_container_width=True)

    with text_col:
        st.markdown(
            f"""
            <div class="caption-box">
                <span>English caption preview</span>
                <p>{escape(caption.english)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"""
            <div class="caption-box">
                <span>Korean explanation preview</span>
                <p>{escape(caption.korean)}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        pipeline = pd.DataFrame(
            [
                {"Step": "1. Image", "Model / Logic": "이미지 입력", "Output": "시각 정보"},
                {"Step": "2. BLIP", "Model / Logic": "Vision encoder + language decoder", "Output": "영어 캡션"},
                {"Step": "3. BLEU", "Model / Logic": "생성 문장과 기준 문장 비교", "Output": "평가 점수"},
                {"Step": "4. Translation", "Model / Logic": "번역 모델 또는 후처리", "Output": "한국어 설명"},
            ]
        )
        st.markdown("##### BLIP -> BLEU -> Translation")
        st.dataframe(pipeline, hide_index=True, use_container_width=True)

    st.markdown("##### 샘플 이미지 보드")
    cols = st.columns(6)
    for index, image in enumerate(images[:12]):
        with cols[index % 6]:
            st.image(str(image), caption=image.name.replace("COCO_val2014_", ""), use_container_width=True)


def main() -> None:
    render_sidebar()
    render_overview()

    tab_movie, tab_fashion, tab_caption = st.tabs(
        ["Lab 1 영화 리뷰 분석", "Lab 6 패션 검색", "Lab 9 이미지 캡셔닝"]
    )
    with tab_movie:
        render_movie_lab()
    with tab_fashion:
        render_fashion_lab()
    with tab_caption:
        render_caption_lab()


if __name__ == "__main__":
    main()
