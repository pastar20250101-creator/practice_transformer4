from __future__ import annotations

import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class LabCard:
    lab_id: str
    title: str
    subtitle: str
    study_focus: str
    artifact: str


@dataclass(frozen=True)
class MovieReview:
    movie_title: str
    release_year: str
    review_text: str
    rating: int
    sentiment: str
    source: str


@dataclass(frozen=True)
class FashionLook:
    title: str
    category: str
    image: str
    tags: tuple[str, ...]
    palette: str
    styling_note: str
    source: str


@dataclass(frozen=True)
class CaptionPair:
    english: str
    korean: str


LOCAL_FASHION_DIR = Path("lab6_clip_multimodal_fashion_search") / "fashion_images"

STOPWORDS = {
    "영화",
    "작품",
    "장면",
    "느낌",
    "정도",
    "너무",
    "정말",
    "조금",
    "보고",
    "보는",
    "있다",
    "했다",
    "좋다",
    "좋고",
    "하지만",
    "그리고",
    "이라",
    "으로",
    "해서",
}


REMOTE_FASHION_LOOKS = [
    (
        "Berlin Street Hoodie",
        "streetwear",
        "https://images.unsplash.com/photo-1523398002811-999ca8dec234?auto=format&fit=crop&w=900&q=80",
        ("street", "hoodie", "black", "urban", "casual"),
        "black / concrete",
        "후디와 와이드 팬츠로 도시적인 실루엣을 만든 스트릿 룩",
    ),
    (
        "Minimal City Neutral",
        "minimal",
        "https://images.unsplash.com/photo-1485968579580-b6d095142e6e?auto=format&fit=crop&w=900&q=80",
        ("minimal", "neutral", "city", "wide pants", "bag"),
        "white / black / camel",
        "색을 줄이고 핏과 가방 포인트로 완성한 미니멀 스타일",
    ),
    (
        "Editorial Blue Dress",
        "dress",
        "https://images.unsplash.com/photo-1496747611176-843222e1e57c?auto=format&fit=crop&w=900&q=80",
        ("dress", "editorial", "blue", "formal", "runway"),
        "navy / pink",
        "볼륨감 있는 드레스로 한 장면을 강하게 만드는 에디토리얼 룩",
    ),
    (
        "White Sneaker Studio",
        "sneakers",
        "https://images.unsplash.com/photo-1549298916-b41d501d3772?auto=format&fit=crop&w=900&q=80",
        ("sneaker", "white", "studio", "product", "clean"),
        "white / cream",
        "제품 사진처럼 신발 디테일을 선명하게 보여주는 스니커즈 컷",
    ),
    (
        "Tailored Office Layer",
        "office",
        "https://images.unsplash.com/photo-1503342217505-b0a15ec3261c?auto=format&fit=crop&w=900&q=80",
        ("office", "tailored", "jacket", "clean", "formal"),
        "charcoal / ivory",
        "재킷과 단정한 레이어로 발표 자료에 어울리는 오피스 룩",
    ),
    (
        "Leather Boot Texture",
        "sneakers",
        "https://images.unsplash.com/photo-1608256246200-53e635b5b65f?auto=format&fit=crop&w=900&q=80",
        ("boots", "leather", "brown", "texture", "fall"),
        "brown / black",
        "가죽 질감과 묵직한 색감으로 계절감을 만드는 부츠 스타일",
    ),
    (
        "City Trench Coat",
        "outer",
        "https://images.unsplash.com/photo-1529139574466-a303027c1d8b?auto=format&fit=crop&w=900&q=80",
        ("outer", "coat", "trench", "city", "classic"),
        "beige / black",
        "긴 코트로 전체 비율을 잡아주는 클래식 아우터 룩",
    ),
    (
        "Denim Weekend Fit",
        "casual",
        "https://images.unsplash.com/photo-1515886657613-9f3515b0c78f?auto=format&fit=crop&w=900&q=80",
        ("casual", "denim", "weekend", "blue", "street"),
        "denim / white",
        "데님 중심으로 편하고 선명한 주말 스타일을 만드는 룩",
    ),
    (
        "Monochrome Bag Detail",
        "accessories",
        "https://images.unsplash.com/photo-1590874103328-eac38a683ce7?auto=format&fit=crop&w=900&q=80",
        ("bag", "accessory", "monochrome", "detail", "minimal"),
        "black / silver",
        "룩 전체보다 가방과 소재 디테일에 집중하는 액세서리 컷",
    ),
    (
        "Soft Knit Layer",
        "minimal",
        "https://images.unsplash.com/photo-1516762689617-e1cffcef479d?auto=format&fit=crop&w=900&q=80",
        ("knit", "soft", "layer", "warm", "neutral"),
        "cream / gray",
        "부드러운 니트와 레이어링으로 안정적인 무드를 만드는 룩",
    ),
]


LOCAL_FASHION_LOOKS = [
    ("화이트 스니커즈", "sneakers", ("sneaker", "white", "clean", "product")),
    ("클래식 셔츠", "office", ("shirt", "classic", "office", "minimal")),
    ("캐주얼 티셔츠", "casual", ("tshirt", "casual", "daily", "basic")),
    ("우아한 드레스", "dress", ("dress", "elegant", "formal", "evening")),
    ("와이드 팬츠", "minimal", ("wide pants", "minimal", "neutral", "office")),
    ("오버사이즈 아우터", "outer", ("outer", "oversized", "street", "layer")),
    ("슬림 핏 바지", "office", ("pants", "slim", "tailored", "formal")),
    ("빈티지 선글라스", "accessories", ("sunglasses", "vintage", "accessory", "summer")),
    ("부츠", "sneakers", ("boots", "leather", "fall", "black")),
    ("미니 크로스백", "accessories", ("bag", "crossbody", "mini", "detail")),
    ("미니 스커트", "casual", ("skirt", "mini", "casual", "youth")),
    ("롱 드레스", "dress", ("dress", "long", "elegant", "evening")),
]


def lab_summary_cards() -> list[LabCard]:
    return [
        LabCard(
            lab_id="Lab 1",
            title="Real movie review analysis",
            subtitle="실제 영화명을 기준으로 리뷰 CSV를 분석하는 텍스트 대시보드",
            study_focus="전처리, 토큰화, 불용어 제거, 감성 비율, 키워드 빈도",
            artifact="movie_reviews.csv, sentiment metrics, keyword charts",
        ),
        LabCard(
            lab_id="Lab 6",
            title="Fashion multimodal search",
            subtitle="실제 의상 사진과 로컬 이미지를 함께 쓰는 패션 검색 보드",
            study_focus="이미지와 텍스트를 같은 의미 공간에서 비교하는 CLIP 개념",
            artifact="20+ outfit images, categories, tags, search ranking",
        ),
        LabCard(
            lab_id="Lab 9",
            title="Image captioning studio",
            subtitle="이미지를 선택하고 영어/한국어 캡션 미리보기를 확인",
            study_focus="Vision encoder, language decoder, caption generation, BLEU",
            artifact="COCO/user images, BLIP pipeline table, caption previews",
        ),
    ]


def load_movie_reviews(csv_path: Path) -> list[MovieReview]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        return [
            MovieReview(
                movie_title=row["movie_title"].strip(),
                release_year=row["release_year"].strip(),
                review_text=row["review_text"].strip(),
                rating=int(row["rating"]),
                sentiment=row["sentiment"].strip(),
                source=row["source"].strip(),
            )
            for row in rows
            if row.get("review_text", "").strip()
        ]


def filter_reviews(reviews: Iterable[MovieReview], movie_title: str) -> list[MovieReview]:
    review_list = list(reviews)
    if movie_title == "전체":
        return review_list
    return [review for review in review_list if review.movie_title == movie_title]


def review_summary(reviews: Iterable[MovieReview], movie_title: str) -> dict[str, float | int]:
    selected = filter_reviews(reviews, movie_title)
    if not selected:
        return {
            "review_count": 0,
            "average_rating": 0.0,
            "average_score": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
        }

    positives = sum(1 for review in selected if review.sentiment == "positive")
    negatives = sum(1 for review in selected if review.sentiment == "negative")
    average_rating = sum(review.rating for review in selected) / len(selected)
    return {
        "review_count": len(selected),
        "average_rating": round(average_rating, 2),
        "average_score": round(positives / len(selected), 3),
        "positive_ratio": round(positives / len(selected), 3),
        "negative_ratio": round(negatives / len(selected), 3),
    }


def tokenize_korean_text(text: str) -> list[str]:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    return [
        token
        for token in cleaned.split()
        if len(token) > 1 and token not in STOPWORDS
    ]


def keyword_counts(reviews: Iterable[MovieReview], sentiment: str | None = None) -> Counter[str]:
    counter: Counter[str] = Counter()
    for review in reviews:
        if sentiment is not None and review.sentiment != sentiment:
            continue
        counter.update(tokenize_korean_text(review.review_text))
    return counter


def get_fashion_looks() -> list[FashionLook]:
    looks = [
        FashionLook(
            title=title,
            category=category,
            image=image,
            tags=tags,
            palette=palette,
            styling_note=note,
            source="Unsplash remote image",
        )
        for title, category, image, tags, palette, note in REMOTE_FASHION_LOOKS
    ]

    for title, category, tags in LOCAL_FASHION_LOOKS:
        image_path = LOCAL_FASHION_DIR / f"{title.replace(' ', '_')}.jpg"
        looks.append(
            FashionLook(
                title=title,
                category=category,
                image=str(image_path),
                tags=tags,
                palette="local asset",
                styling_note="실습 6 폴더에 저장된 패션 이미지 자산입니다.",
                source="Local lab asset",
            )
        )
    return looks


def _query_tokens(text: str) -> set[str]:
    normalized = text.lower().replace("_", " ").replace("-", " ").replace("/", " ")
    return {token for token in normalized.split() if token}


def _fashion_score(query_tokens: set[str], look: FashionLook) -> tuple[int, int, str]:
    haystack = " ".join((look.title, look.category, look.palette, look.styling_note, *look.tags)).lower()
    exact_hits = sum(1 for token in query_tokens if token in look.tags)
    soft_hits = sum(1 for token in query_tokens if token in haystack)
    return (exact_hits, soft_hits, look.title)


def search_fashion_looks(query: str, looks: Iterable[FashionLook], limit: int = 8) -> list[FashionLook]:
    look_list = list(looks)
    query_tokens = _query_tokens(query)
    if not query_tokens:
        return look_list[:limit]
    return sorted(look_list, key=lambda look: _fashion_score(query_tokens, look), reverse=True)[:limit]


def describe_lab9_image(image_name: str) -> CaptionPair:
    if image_name == "my_photo.png":
        return CaptionPair(
            english="A personal image selected for image captioning practice, ready for a BLIP-style caption preview.",
            korean="사용자가 직접 추가한 이미지입니다. 이미지 캡셔닝 모델이 장면을 읽고 설명 문장으로 바꾸는 과정을 보여줍니다.",
        )

    image_id = Path(image_name).stem.replace("COCO_val2014_", "")
    return CaptionPair(
        english=f"COCO validation image {image_id}: a caption-generation sample for showing how visual features become natural language.",
        korean=f"COCO 검증 이미지 {image_id}입니다. 이미지 특징이 자연어 캡션으로 바뀌는 과정을 보여주는 샘플입니다.",
    )
