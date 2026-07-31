from pathlib import Path
import unittest

from app_utils import (
    describe_lab9_image,
    load_fashion_items,
    load_lab9_images,
    search_fashion_items,
)


class AppUtilsTest(unittest.TestCase):
    def test_load_fashion_items_finds_local_assets(self):
        items = load_fashion_items(Path("lab6_clip_multimodal_fashion_search/fashion_images"))

        self.assertEqual(len(items), 12)
        self.assertTrue(all(item.path.exists() for item in items))
        self.assertTrue(any("스니커즈" in item.label for item in items))

    def test_search_fashion_items_prioritizes_matching_label(self):
        items = load_fashion_items(Path("lab6_clip_multimodal_fashion_search/fashion_images"))

        results = search_fashion_items("화이트 스니커즈", items, limit=3)

        self.assertEqual(results[0].label, "화이트 스니커즈")
        self.assertEqual(len(results), 3)

    def test_load_lab9_images_includes_coco_and_user_photo(self):
        images = load_lab9_images(Path("lab9_image_captioning"))

        self.assertTrue(any(image.path.name == "my_photo.png" for image in images))
        self.assertTrue(any(image.path.name.startswith("COCO_val2014") for image in images))

    def test_describe_lab9_image_returns_caption_pair(self):
        image = load_lab9_images(Path("lab9_image_captioning"))[0]

        caption = describe_lab9_image(image)

        self.assertTrue(caption.english)
        self.assertTrue(caption.korean)


if __name__ == "__main__":
    unittest.main()
