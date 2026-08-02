from __future__ import annotations

import json
import re
from io import BytesIO
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from PIL import Image


LAB1_DIR = Path("lab1_korean_movie_review_analysis")
LAB2_DIR = Path("lab2_seq2seq_translation_modeling")
LAB6_DIR = Path("lab6_clip_multimodal_fashion_search")
LAB9_DIR = Path("lab9_image_captioning")

KOREAN_STOPWORDS = {
    "영화",
    "정말",
    "너무",
    "그리고",
    "하지만",
    "그냥",
    "있는",
    "없는",
    "하는",
    "했다",
    "입니다",
    "같은",
    "이런",
    "저런",
    "보고",
    "봤는데",
}

POSITIVE_WORDS = {
    "좋",
    "최고",
    "감동",
    "재미",
    "훌륭",
    "몰입",
    "추천",
    "명작",
    "완벽",
    "인상",
    "멋",
    "웃",
}

NEGATIVE_WORDS = {
    "별로",
    "최악",
    "지루",
    "실망",
    "아쉽",
    "쓰레기",
    "부족",
    "허술",
    "노잼",
    "억지",
    "불편",
    "낭비",
}

_KOREAN_COLOR_NAMES = {
    "black": "검은색",
    "blue": "파란색",
    "gray": "회색",
    "green": "초록색",
    "orange": "주황색",
    "pink": "분홍색",
    "purple": "보라색",
    "red": "붉은색",
    "white": "흰색",
    "yellow": "노란색",
}


@dataclass(frozen=True)
class NotebookSummary:
    lab_id: str
    title: str
    area: str
    notebook_path: Path
    goal: str
    pipeline: tuple[str, ...]


@dataclass(frozen=True)
class NotebookStats:
    cells: int
    markdown_cells: int
    code_cells: int
    output_cells: int


@dataclass(frozen=True)
class MovieReviewSample:
    movie: str
    review: str


@dataclass(frozen=True)
class TokenRecord:
    rank: int
    word: str
    count: int
    percent: float | None


@dataclass(frozen=True)
class Lab1Artifact:
    title: str
    path: Path
    note: str


@dataclass(frozen=True)
class TranslationPair:
    korean: str
    english: str


@dataclass(frozen=True)
class FashionProfile:
    name: str
    category: str
    description: str
    image_path: Path
    query: str
    tags: tuple[str, ...]


@dataclass(frozen=True)
class CaptionRecord:
    image_name: str
    image_path: Path
    english_captions: tuple[str, ...]
    korean_captions: tuple[str, ...]


@dataclass(frozen=True)
class UploadedCaptionPreview:
    english: str
    korean: str
    width: int
    height: int
    orientation: str
    dominant_tones: tuple[str, ...]


MOVIE_REVIEW_SAMPLES = (
    MovieReviewSample(
        movie="파묘",
        review="분위기가 강렬하고 배우들의 연기가 좋았다. 초반 몰입감이 뛰어나서 끝까지 집중했다.",
    ),
    MovieReviewSample(
        movie="서울의 봄",
        review="긴장감이 높고 장면 전환이 탄탄했다. 역사적 사건을 묵직하게 보여줘서 인상적이었다.",
    ),
    MovieReviewSample(
        movie="범죄도시4",
        review="액션은 시원하고 속도감이 좋았다. 다만 이야기는 익숙해서 신선함은 조금 부족했다.",
    ),
    MovieReviewSample(
        movie="탈주",
        review="추격 장면은 좋고 배우들의 호흡도 괜찮았다. 중반 전개가 조금 아쉬웠지만 몰입은 됐다.",
    ),
    MovieReviewSample(
        movie="베테랑2",
        review="타격감 있는 액션과 빠른 전개가 인상적이었다. 캐릭터의 매력도 살아 있어서 재미있었다.",
    ),
    MovieReviewSample(
        movie="파일럿",
        review="가볍게 웃기 좋은 코미디였다. 몇몇 장면은 예상 가능했지만 배우들의 리듬이 좋았다.",
    ),
)


def notebook_summaries() -> list[NotebookSummary]:
    return [
        NotebookSummary(
            lab_id="실습1",
            title="한국어 영화 리뷰 분석 및 시각화",
            area="텍스트 전처리 · 감성/빈도 시각화",
            notebook_path=LAB1_DIR / "lab1_korean_movie_review_analysis.ipynb",
            goal="한국어 리뷰를 정규표현식과 형태소 분석으로 전처리하고, 빈도 분석 결과를 시각화합니다.",
            pipeline=("입력 파일 로드", "정규표현식 전처리", "Okt 형태소 분석", "불용어/품사 필터링", "BarPlot + WordCloud"),
        ),
        NotebookSummary(
            lab_id="실습2",
            title="Seq2Seq 기반 번역 AI 모델링",
            area="인코더-디코더 번역 모델",
            notebook_path=LAB2_DIR / "lab2_seq2seq_translation_modeling.ipynb",
            goal="한국어 문장을 영어로 번역하는 Seq2Seq 모델 구조와 학습/추론 흐름을 확인합니다.",
            pipeline=("데이터 로드", "토큰화", "Encoder", "Luong Attention", "Decoder", "Greedy Decoding"),
        ),
        NotebookSummary(
            lab_id="실습6",
            title="CLIP 멀티모달 패션 검색",
            area="텍스트-이미지 교차 검색",
            notebook_path=LAB6_DIR / "lab6_clip_multimodal_fashion_search.ipynb",
            goal="CLIP으로 이미지와 텍스트를 같은 벡터 공간에 올리고 코사인 유사도로 검색합니다.",
            pipeline=("패션 프로필 작성", "이미지 캐시 확인", "텍스트 벡터화", "이미지 벡터화", "코사인 유사도 검색"),
        ),
        NotebookSummary(
            lab_id="실습9",
            title="Image Captioning",
            area="이미지 -> 설명 문장 생성",
            notebook_path=LAB9_DIR / "lab9_image_captioning.ipynb",
            goal="이미지 인코더와 캡션 디코더로 설명 문장을 만들고 MSCOCO 한국어 캡션과 비교합니다.",
            pipeline=("MSCOCO 데이터 탐색", "COCO 이미지 샘플 확인", "BLIP 캡션 생성", "BLEU 평가", "한국어 번역"),
        ),
    ]


def notebook_stats(path: Path) -> NotebookStats:
    notebook = _read_notebook(path)
    cells = notebook.get("cells", [])
    markdown_cells = sum(1 for cell in cells if cell.get("cell_type") == "markdown")
    code_cells = sum(1 for cell in cells if cell.get("cell_type") == "code")
    output_cells = sum(1 for cell in cells if cell.get("outputs"))
    return NotebookStats(
        cells=len(cells),
        markdown_cells=markdown_cells,
        code_cells=code_cells,
        output_cells=output_cells,
    )


def notebook_headings(path: Path, max_items: int = 12) -> list[str]:
    notebook = _read_notebook(path)
    headings: list[str] = []
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") != "markdown":
            continue
        source = "".join(cell.get("source", []))
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                headings.append(stripped)
                if len(headings) >= max_items:
                    return headings
    return headings


def lab1_artifacts() -> list[Lab1Artifact]:
    output_dir = LAB1_DIR / "output"
    return [
        Lab1Artifact(
            title="기본 분석 BarPlot + WordCloud",
            path=output_dir / "기본_분석_BarPlot_+_WordCloud.png",
            note="노트북 Step 2에서 저장된 기본 빈도 분석 시각화입니다.",
        ),
        Lab1Artifact(
            title="개선된 분석 BarPlot + WordCloud",
            path=output_dir / "개선된_분석_BarPlot_+_WordCloud.png",
            note="불용어/필터링을 개선한 뒤 저장된 시각화입니다.",
        ),
        Lab1Artifact(
            title="고급 분석 그리드",
            path=output_dir / "advanced_analysis_grid.png",
            note="상위 단어, 누적 비율, 빈도 분포를 함께 보여주는 고급 분석 이미지입니다.",
        ),
        Lab1Artifact(
            title="품사별 분석 비교",
            path=output_dir / "pos_analysis_comparison.png",
            note="명사, 동사, 형용사 기준의 품사별 분석 비교 이미지입니다.",
        ),
    ]


def load_token_records(path: Path, limit: int = 30) -> list[TokenRecord]:
    records: list[TokenRecord] = []
    pattern = re.compile(r"^\s*(\d+)\.\s+(.+?)\s+([\d,]+)\s+\(\s*([\d.]+)%\)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if not match:
            continue
        records.append(
            TokenRecord(
                rank=int(match.group(1)),
                word=match.group(2).strip(),
                count=int(match.group(3).replace(",", "")),
                percent=float(match.group(4)),
            )
        )
        if len(records) >= limit:
            break
    return records


def lab1_summary_numbers() -> dict[str, str]:
    return {
        "총 토큰 수": "180,918개",
        "기본 고유 단어": "34,530개",
        "개선 고유 단어": "5,465개",
        "감소율": "84.2%",
    }


def movie_review_samples() -> tuple[MovieReviewSample, ...]:
    return MOVIE_REVIEW_SAMPLES


def movie_review_sample_rows(samples: Iterable[MovieReviewSample] | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample in samples or MOVIE_REVIEW_SAMPLES:
        result = analyze_movie_reviews(sample.review)
        rows.append(
            {
                "영화": sample.movie,
                "감성": result["label"],
                "점수": result["score"],
                "긍정": result["positive_hits"],
                "부정": result["negative_hits"],
                "핵심 키워드": ", ".join(result["keywords"][:5]),
                "리뷰": sample.review,
            }
        )
    return rows


def analyze_movie_reviews(text: str) -> dict[str, object]:
    tokens = tokenize_korean_text(text)
    counter = Counter(tokens)
    positive_hits = _count_sentiment_hits(text, POSITIVE_WORDS)
    negative_hits = _count_sentiment_hits(text, NEGATIVE_WORDS)
    score = positive_hits - negative_hits

    if score > 0:
        label = "긍정"
    elif score < 0:
        label = "부정"
    else:
        label = "중립"

    return {
        "label": label,
        "score": score,
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "token_count": len(tokens),
        "unique_count": len(counter),
        "keywords": [word for word, _ in counter.most_common(10)],
        "keyword_counts": counter,
    }


def tokenize_korean_text(text: str) -> list[str]:
    cleaned = re.sub(r"[^0-9A-Za-z가-힣\s]", " ", text)
    return [
        normalized
        for token in cleaned.split()
        for normalized in [_normalize_korean_token(token)]
        if len(normalized) > 1 and normalized not in KOREAN_STOPWORDS
    ]


def load_translation_pairs(limit: int = 12) -> list[TranslationPair]:
    korean_lines = (LAB2_DIR / "data" / "train_kor.txt").read_text(encoding="utf-8").splitlines()
    english_lines = (LAB2_DIR / "data" / "train_eng.txt").read_text(encoding="utf-8").splitlines()
    pairs: list[TranslationPair] = []
    for korean, english in zip(korean_lines, english_lines):
        if korean.strip() and english.strip():
            pairs.append(TranslationPair(korean=korean.strip(), english=english.strip()))
        if len(pairs) >= limit:
            break
    return pairs


def translation_input_text() -> str:
    return (LAB2_DIR / "data" / "input.txt").read_text(encoding="utf-8").strip()


def translation_output_text() -> str:
    return (LAB2_DIR / "output.txt").read_text(encoding="utf-8").strip()


def translate_by_similarity(query: str, pairs: Iterable[TranslationPair]) -> dict[str, object]:
    pair_list = list(pairs)
    query_tokens = _query_tokens(query)
    if not pair_list or not query_tokens:
        return {"source": "", "translation": "", "similarity": 0.0}

    scored = []
    for pair in pair_list:
        source_tokens = _query_tokens(pair.korean)
        union = query_tokens | source_tokens
        intersection = query_tokens & source_tokens
        similarity = len(intersection) / len(union) if union else 0.0
        soft = sum(1 for token in query_tokens if token and token in pair.korean)
        scored.append((similarity, soft, pair))

    similarity, _, best = max(scored, key=lambda item: (item[0], item[1], len(item[2].korean)))
    return {
        "source": best.korean,
        "translation": best.english,
        "similarity": round(similarity, 3),
    }


def fashion_profiles() -> list[FashionProfile]:
    profile_map = {
        "클래식 셔츠": (
            "office",
            "깔끔한 셔츠와 단정한 실루엣으로 오피스룩과 미니멀룩에 잘 맞는다.",
            "classic white shirt office minimal fashion woman",
            ("shirt", "office", "minimal", "white", "classic"),
        ),
        "캐주얼 티셔츠": (
            "casual",
            "편한 코튼 티셔츠와 루즈한 핏으로 일상적인 캐주얼 코디에 어울린다.",
            "casual t-shirt everyday street fashion woman",
            ("tshirt", "casual", "daily", "streetwear", "cotton"),
        ),
        "우아한 드레스": (
            "dress",
            "검은색 계열의 세련된 드레스로 저녁 약속이나 포멀한 자리에 잘 맞는다.",
            "elegant black dress minimal formal woman",
            ("dress", "black", "elegant", "formal", "minimal"),
        ),
        "오버사이즈 아우터": (
            "outer",
            "넉넉한 재킷이나 코트 실루엣으로 스트리트 무드와 레이어드 스타일을 만든다.",
            "oversized jacket coat outer streetwear fashion",
            ("outer", "jacket", "coat", "oversized", "streetwear"),
        ),
        "슬림 핏 바지": (
            "minimal",
            "군더더기 없는 슬림 팬츠로 셔츠, 니트, 재킷과 쉽게 조합된다.",
            "slim fit pants minimal office fashion woman",
            ("pants", "slim", "minimal", "office", "clean"),
        ),
        "와이드 팬츠": (
            "streetwear",
            "넓은 바지 실루엣으로 편안하면서도 트렌디한 스트리트 코디에 어울린다.",
            "wide pants streetwear casual fashion woman",
            ("pants", "wide", "streetwear", "casual", "trend"),
        ),
        "미니 스커트": (
            "casual",
            "짧은 스커트와 가벼운 상의 조합으로 산뜻한 데일리룩을 만들 수 있다.",
            "mini skirt casual summer fashion style",
            ("skirt", "mini", "casual", "summer", "daily"),
        ),
        "롱 드레스": (
            "dress",
            "긴 드레스 실루엣으로 차분하고 우아한 분위기를 강조한다.",
            "long dress elegant woman formal fashion",
            ("dress", "long", "elegant", "formal", "classic"),
        ),
        "화이트 스니커즈": (
            "sneakers",
            "깨끗한 흰색 스니커즈로 캐주얼, 오피스, 미니멀 코디에 모두 활용된다.",
            "white sneakers casual minimal street fashion",
            ("sneakers", "white", "shoes", "casual", "minimal"),
        ),
        "부츠": (
            "sneakers",
            "긴 부츠로 겨울 아우터나 드레스 코디에 강한 포인트를 준다.",
            "boots winter shoes fashion woman",
            ("boots", "shoes", "winter", "outer", "dress"),
        ),
        "미니 크로스백": (
            "accessories",
            "작고 단정한 크로스백으로 미니멀한 착장에 포인트를 더한다.",
            "mini crossbody bag minimal accessory fashion",
            ("bag", "crossbody", "accessories", "minimal", "daily"),
        ),
        "빈티지 선글라스": (
            "accessories",
            "레트로한 선글라스로 여름 코디나 스트리트룩에 개성을 더한다.",
            "vintage sunglasses accessory summer streetwear fashion",
            ("sunglasses", "accessories", "vintage", "summer", "streetwear"),
        ),
    }
    image_dir = LAB6_DIR / "fashion_images"
    return [
        FashionProfile(
            name=name,
            category=category,
            description=description,
            image_path=image_dir / f"{name.replace(' ', '_')}.jpg",
            query=query,
            tags=tags,
        )
        for name, (category, description, query, tags) in profile_map.items()
    ]


def search_fashion_profiles(query: str, profiles: Iterable[FashionProfile], limit: int = 12) -> list[FashionProfile]:
    profile_list = list(profiles)
    query_tokens = _query_tokens(query)
    if not query_tokens:
        return profile_list[:limit]
    return sorted(profile_list, key=lambda profile: _profile_score(query_tokens, profile), reverse=True)[:limit]


def local_coco_images() -> list[Path]:
    return sorted((LAB9_DIR / "coco_images").glob("*.jpg"))


@lru_cache(maxsize=1)
def load_caption_records() -> tuple[CaptionRecord, ...]:
    image_paths = {path.name: path for path in local_coco_images()}
    if not image_paths:
        return ()

    caption_path = LAB9_DIR / "korean_image_captioning_dataset" / "MSCOCO_train_val_Korean.json"
    with caption_path.open("r", encoding="utf-8") as file:
        coco_data = json.load(file)

    records: list[CaptionRecord] = []
    for item in coco_data:
        image_name = Path(item.get("file_path", "")).name
        if image_name not in image_paths:
            continue
        records.append(
            CaptionRecord(
                image_name=image_name,
                image_path=image_paths[image_name],
                english_captions=tuple(item.get("captions", [])),
                korean_captions=tuple(item.get("caption_ko", [])),
            )
        )

    return tuple(sorted(records, key=lambda record: record.image_name))


def my_photo_path() -> Path:
    return LAB9_DIR / "my_photo.png"


def my_photo_caption_record() -> CaptionRecord:
    return CaptionRecord(
        image_name="my_photo.png",
        image_path=my_photo_path(),
        english_captions=(
            "A young woman poses on a sunny beach beside SpongeBob SquarePants and Mr. Krabs character statues.",
            "The photo shows a seaside display with a Krusty Krab sign, colorful beach balls, and a bridge in the background.",
            "A person is standing on sand next to large cartoon figures under a clear blue sky.",
            "The image captures a playful beach attraction featuring SpongeBob-themed decorations and visitors by the water.",
            "A bright outdoor travel photo shows a woman making a peace sign near character sculptures on the beach.",
        ),
        korean_captions=(
            "맑은 하늘 아래 해변에서 한 여성이 스폰지밥과 집게사장 캐릭터 조형물 옆에서 포즈를 취하고 있습니다.",
            "사진에는 크러스티 크랩 표지판, 알록달록한 비치볼 장식, 그리고 뒤쪽의 바다와 다리가 함께 보입니다.",
            "모래사장 위에서 사람이 커다란 만화 캐릭터 조형물 옆에 서 있는 장면입니다.",
            "스폰지밥 테마 장식과 해변 방문객들이 어우러진 밝고 활기찬 관광지 사진입니다.",
            "여성이 캐릭터 조형물 옆에서 브이 포즈를 취한 야외 여행 사진입니다.",
        ),
    )


def caption_uploaded_image(image_bytes: bytes) -> UploadedCaptionPreview:
    with Image.open(BytesIO(image_bytes)) as image:
        rgb_image = image.convert("RGB")
        width, height = rgb_image.size
        orientation, orientation_kr = _image_orientation(width, height)
        brightness, brightness_kr = _image_brightness_label(rgb_image)
        dominant_tones = _dominant_color_tones(rgb_image)

    english_tones = ", ".join(dominant_tones)
    korean_tones = ", ".join(_KOREAN_COLOR_NAMES.get(tone, tone) for tone in dominant_tones)

    english = (
        f"A {orientation} uploaded image at {width}x{height}, "
        f"with {brightness} lighting and dominant {english_tones} tones."
    )
    korean = (
        f"{orientation_kr} 업로드 이미지입니다. "
        f"{width}x{height} 해상도이며 {brightness_kr} 밝기와 {korean_tones} 계열 색감이 두드러집니다."
    )

    return UploadedCaptionPreview(
        english=english,
        korean=korean,
        width=width,
        height=height,
        orientation=orientation,
        dominant_tones=dominant_tones,
    )


def _read_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _query_tokens(query: str) -> set[str]:
    normalized = query.lower().replace("_", " ").replace("-", " ")
    return {token for token in re.split(r"\s+", normalized) if token}


def _count_sentiment_hits(text: str, lexicon: set[str]) -> int:
    normalized = text.lower()
    return sum(normalized.count(word.lower()) for word in lexicon)


def _image_orientation(width: int, height: int) -> tuple[str, str]:
    if width > height * 1.08:
        return "landscape", "가로형"
    if height > width * 1.08:
        return "portrait", "세로형"
    return "square", "정방형"


def _image_brightness_label(image: Image.Image) -> tuple[str, str]:
    thumbnail = image.copy()
    thumbnail.thumbnail((80, 80))
    pixels = _image_pixels(thumbnail)
    average = sum((red + green + blue) / 3 for red, green, blue in pixels) / max(len(pixels), 1)
    if average >= 175:
        return "bright", "밝은"
    if average <= 80:
        return "dark", "어두운"
    return "balanced", "중간"


def _dominant_color_tones(image: Image.Image, limit: int = 3) -> tuple[str, ...]:
    thumbnail = image.copy()
    thumbnail.thumbnail((80, 80))
    tone_counts = Counter(_classify_rgb_tone(red, green, blue) for red, green, blue in _image_pixels(thumbnail))
    return tuple(tone for tone, _ in tone_counts.most_common(limit))


def _image_pixels(image: Image.Image) -> list[tuple[int, int, int]]:
    flattened_data = getattr(image, "get_flattened_data", None)
    if flattened_data is not None:
        return list(flattened_data())
    return list(image.getdata())


def _classify_rgb_tone(red: int, green: int, blue: int) -> str:
    brightness = (red + green + blue) / 3
    spread = max(red, green, blue) - min(red, green, blue)
    if spread < 28:
        if brightness >= 205:
            return "white"
        if brightness <= 65:
            return "black"
        return "gray"
    if red >= 170 and green >= 135 and blue < 120:
        return "yellow"
    if red >= 170 and green >= 80 and blue < 95:
        return "orange"
    if red >= green + 35 and red >= blue + 35:
        return "red"
    if green >= red + 30 and green >= blue + 25:
        return "green"
    if blue >= red + 30 and blue >= green + 20:
        return "blue"
    if red >= 145 and blue >= 135 and green < 135:
        return "purple"
    if red >= 180 and blue >= 120 and green < 160:
        return "pink"
    return "gray"


def _normalize_korean_token(token: str) -> str:
    normalized = token.strip()
    for suffix in ("이었다", "였다", "이다", "입니다"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            normalized = normalized[: -len(suffix)]
            break
    for suffix in ("으로", "에서", "에게", "까지", "부터", "보다", "처럼", "이라", "가", "이", "은", "는", "을", "를", "도", "만"):
        if normalized.endswith(suffix) and len(normalized) > len(suffix) + 1:
            normalized = normalized[: -len(suffix)]
            break
    return normalized


def _profile_score(query_tokens: set[str], profile: FashionProfile) -> tuple[int, int, str]:
    haystack = " ".join((profile.name, profile.category, profile.description, profile.query, *profile.tags)).lower()
    exact = sum(1 for token in query_tokens if token in profile.name.lower())
    soft = sum(1 for token in query_tokens if token in haystack)
    return (exact, soft, profile.name)
