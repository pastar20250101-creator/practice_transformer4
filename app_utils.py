from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DemoImage:
    path: Path
    label: str


@dataclass(frozen=True)
class CaptionPair:
    english: str
    korean: str


def label_from_path(path: Path) -> str:
    return path.stem.replace("_", " ")


def load_fashion_items(image_dir: Path) -> list[DemoImage]:
    paths = sorted(image_dir.glob("*.jpg"), key=lambda path: label_from_path(path))
    return [DemoImage(path=path, label=label_from_path(path)) for path in paths]


def _tokens(text: str) -> set[str]:
    normalized = text.lower().replace("_", " ").replace("-", " ")
    return {token for token in normalized.split() if token}


def _score_query(query_tokens: set[str], item: DemoImage) -> tuple[int, int, str]:
    label_tokens = _tokens(item.label)
    overlap = len(query_tokens & label_tokens)
    substring_bonus = 1 if query_tokens and any(token in item.label.lower() for token in query_tokens) else 0
    return (overlap, substring_bonus, item.label)


def search_fashion_items(query: str, items: Iterable[DemoImage], limit: int = 6) -> list[DemoImage]:
    item_list = list(items)
    query_tokens = _tokens(query)
    if not query_tokens:
        return item_list[:limit]

    ranked = sorted(
        item_list,
        key=lambda item: _score_query(query_tokens, item),
        reverse=True,
    )
    return ranked[:limit]


def load_lab9_images(lab9_dir: Path) -> list[DemoImage]:
    images: list[Path] = []
    user_photo = lab9_dir / "my_photo.png"
    if user_photo.exists():
        images.append(user_photo)
    images.extend(sorted((lab9_dir / "coco_images").glob("*.jpg")))
    return [DemoImage(path=path, label=label_from_path(path)) for path in images]


def describe_lab9_image(image: DemoImage) -> CaptionPair:
    if image.path.name == "my_photo.png":
        return CaptionPair(
            english="A user-provided image for practicing image captioning.",
            korean="이미지 캡셔닝 실습을 위한 사용자 업로드 예시 이미지입니다.",
        )

    image_id = image.path.stem.replace("COCO_val2014_", "")
    return CaptionPair(
        english=f"COCO validation image {image_id}. In the notebook, BLIP generates a natural-language caption for this image.",
        korean=f"COCO 검증 이미지 {image_id}입니다. 노트북에서는 BLIP이 이 이미지를 보고 자연어 설명을 생성합니다.",
    )


def lab_summary_cards() -> list[tuple[str, str, str]]:
    return [
        ("Lab 6", "CLIP fashion search", "Text and image data are compared through vector similarity."),
        ("Lab 9", "Image captioning", "BLIP turns images into captions, then translation can convert them to Korean."),
    ]
