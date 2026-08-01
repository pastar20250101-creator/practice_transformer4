from pathlib import Path
import unittest

from app_utils import (
    describe_lab9_image,
    get_fashion_looks,
    keyword_counts,
    lab_summary_cards,
    load_movie_reviews,
    review_summary,
    search_fashion_looks,
)


ROOT = Path(__file__).resolve().parents[1]


class AppUtilsTest(unittest.TestCase):
    def test_lab_summary_cards_only_cover_lab_1_6_9(self):
        cards = lab_summary_cards()

        self.assertEqual([card.lab_id for card in cards], ["Lab 1", "Lab 6", "Lab 9"])
        self.assertFalse(any("Lab 2" in card.lab_id or "Seq2Seq" in card.title for card in cards))

    def test_streamlit_app_does_not_render_lab2_or_seq2seq(self):
        app_source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")

        self.assertNotIn("Lab 2", app_source)
        self.assertNotIn("Seq2Seq", app_source)

    def test_load_movie_reviews_reads_real_movie_review_snapshot(self):
        reviews = load_movie_reviews(ROOT / "data" / "movie_reviews.csv")

        self.assertGreaterEqual(len(reviews), 24)
        self.assertGreaterEqual(len({review.movie_title for review in reviews}), 4)
        self.assertTrue(all(review.review_text for review in reviews))
        self.assertTrue(all(review.source for review in reviews))

    def test_review_summary_computes_sentiment_metrics(self):
        reviews = load_movie_reviews(ROOT / "data" / "movie_reviews.csv")
        summary = review_summary(reviews, "전체")

        self.assertEqual(summary["review_count"], len(reviews))
        self.assertGreater(summary["positive_ratio"], 0)
        self.assertGreater(summary["negative_ratio"], 0)
        self.assertGreaterEqual(summary["average_score"], 0)
        self.assertLessEqual(summary["average_score"], 1)

    def test_keyword_counts_can_split_positive_and_negative_reviews(self):
        reviews = load_movie_reviews(ROOT / "data" / "movie_reviews.csv")

        positive_keywords = keyword_counts(reviews, sentiment="positive")
        negative_keywords = keyword_counts(reviews, sentiment="negative")

        self.assertGreaterEqual(len(positive_keywords), 5)
        self.assertGreaterEqual(len(negative_keywords), 5)
        self.assertNotEqual(positive_keywords.most_common(1), negative_keywords.most_common(1))

    def test_fashion_looks_are_diverse_and_searchable(self):
        looks = get_fashion_looks()
        local_looks = [look for look in looks if not look.image.startswith("https://")]

        self.assertGreaterEqual(len(looks), 20)
        self.assertGreaterEqual(len({look.category for look in looks}), 8)
        self.assertTrue(any(look.image.startswith("https://") for look in looks))
        self.assertTrue(local_looks)
        self.assertTrue(all((ROOT / look.image).exists() for look in local_looks))

        results = search_fashion_looks("black street sneaker", looks, limit=4)

        self.assertEqual(len(results), 4)
        self.assertTrue(any("street" in " ".join(result.tags) for result in results))

    def test_describe_lab9_image_returns_caption_pair(self):
        caption = describe_lab9_image("COCO_val2014_000000570107.jpg")

        self.assertIn("caption", caption.english.lower())
        self.assertIn("이미지", caption.korean)


if __name__ == "__main__":
    unittest.main()
