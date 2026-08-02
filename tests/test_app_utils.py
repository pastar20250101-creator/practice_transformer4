from pathlib import Path
from io import BytesIO
import subprocess
import unittest

from PIL import Image

from app_utils import (
    analyze_movie_reviews,
    caption_uploaded_image,
    fashion_profiles,
    lab1_artifacts,
    load_caption_records,
    load_token_records,
    load_translation_pairs,
    my_photo_caption_record,
    movie_review_sample_rows,
    movie_review_samples,
    notebook_stats,
    notebook_summaries,
    search_fashion_profiles,
    translate_by_similarity,
    translation_input_text,
    translation_output_text,
)


ROOT = Path(__file__).resolve().parents[1]


class AppUtilsTest(unittest.TestCase):
    def test_streamlit_reflects_available_notebook_labs(self):
        summaries = notebook_summaries()

        self.assertEqual([summary.lab_id for summary in summaries], ["실습1", "실습2", "실습6", "실습9"])
        self.assertTrue(all((ROOT / summary.notebook_path).exists() for summary in summaries))

    def test_streamlit_app_hides_assignment_guide_language(self):
        source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")

        self.assertNotIn("가이드라인 반영", source)
        self.assertNotIn("제출 가이드 기준", source)
        self.assertNotIn("assignment_guide_rows", source)
        self.assertNotIn("st.sidebar", source)
        self.assertNotIn("노트북 목차", source)
        self.assertNotIn("노트북 상태", source)
        self.assertNotIn("노트북", source)
        self.assertNotIn("output 이미지", source)
        self.assertNotIn("st.info", source)

    def test_streamlit_app_has_four_clean_deployment_tabs(self):
        source = (ROOT / "streamlit_app.py").read_text(encoding="utf-8")

        self.assertIn("영화 리뷰 분석기", source)
        self.assertIn("번역기", source)
        self.assertIn("패션 검색기", source)
        self.assertIn("이미지 캡셔닝", source)
        self.assertIn("st.tabs", source)
        self.assertNotIn("render_overview", source)
        self.assertNotIn("render_sidebar", source)

    def test_movie_review_analyzer_returns_sentiment_and_keywords(self):
        result = analyze_movie_reviews("연기가 좋고 스토리가 최고였다. 감동적인 영화였다.")

        self.assertEqual(result["label"], "긍정")
        self.assertGreater(result["score"], 0)
        self.assertIn("연기", result["keywords"])

    def test_movie_review_samples_build_dashboard_rows(self):
        samples = movie_review_samples()
        rows = movie_review_sample_rows(samples)

        self.assertGreaterEqual(len(samples), 4)
        self.assertEqual(len(rows), len(samples))
        self.assertTrue(all(row["영화"] and row["리뷰"] for row in rows))
        self.assertTrue(any(row["감성"] == "긍정" for row in rows))

    def test_retrieval_translator_returns_matching_english_sentence(self):
        pairs = load_translation_pairs(limit=20)
        result = translate_by_similarity("당신은 과일을 따기도 하고 농장 일을 돕게 됩니다.", pairs)

        self.assertIn("fruit", result["translation"].lower())
        self.assertGreater(result["similarity"], 0)

    def test_notebook_stats_reads_ipynb_structure(self):
        for summary in notebook_summaries():
            stats = notebook_stats(ROOT / summary.notebook_path)

            self.assertGreater(stats.cells, 0)
            self.assertGreater(stats.markdown_cells, 0)
            self.assertGreater(stats.code_cells, 0)

    def test_lab1_artifacts_and_tokens_exist(self):
        artifacts = lab1_artifacts()
        basic_tokens = load_token_records(ROOT / "lab1_korean_movie_review_analysis" / "output" / "basic_top_tokens.txt")

        self.assertEqual(len(artifacts), 4)
        self.assertTrue(all((ROOT / artifact.path).exists() for artifact in artifacts))
        self.assertGreaterEqual(len(basic_tokens), 20)
        self.assertEqual(basic_tokens[0].word, "연기")

    def test_lab2_translation_files_are_available(self):
        pairs = load_translation_pairs(limit=5)

        self.assertEqual(len(pairs), 5)
        self.assertIn("Test", translation_input_text())
        self.assertIn("fruit", translation_output_text())
        self.assertTrue(all(pair.korean and pair.english for pair in pairs))

    def test_lab2_translation_data_is_not_ignored_for_cloud_deploy(self):
        result = subprocess.run(
            ["git", "check-ignore", "-q", "lab2_seq2seq_translation_modeling/data/train_kor.txt"],
            cwd=ROOT,
            check=False,
        )

        self.assertNotEqual(result.returncode, 0)

    def test_lab6_profiles_use_the_twelve_notebook_images(self):
        profiles = fashion_profiles()

        self.assertEqual(len(profiles), 12)
        self.assertTrue(all((ROOT / profile.image_path).exists() for profile in profiles))
        self.assertTrue(all(profile.category and profile.tags for profile in profiles))
        self.assertIn("office", {profile.category for profile in profiles})

        results = search_fashion_profiles("우아하고 세련된 검은색 옷", profiles, limit=3)

        self.assertEqual(len(results), 3)
        self.assertTrue(any("드레스" in result.name for result in results))

    def test_lab9_caption_records_match_local_coco_images(self):
        records = load_caption_records()

        self.assertGreaterEqual(len(records), 45)
        self.assertTrue(all((ROOT / record.image_path).exists() for record in records))
        self.assertTrue(all(record.korean_captions for record in records[:5]))
        self.assertTrue(all(record.english_captions for record in records[:5]))

    def test_my_photo_has_actual_scene_captions(self):
        record = my_photo_caption_record()

        self.assertEqual(record.image_name, "my_photo.png")
        self.assertIn("beach", record.english_captions[0].lower())
        self.assertIn("spongebob", record.english_captions[0].lower())
        self.assertIn("해변", record.korean_captions[0])
        self.assertIn("스폰지밥", record.korean_captions[0])

    def test_uploaded_image_caption_describes_image_properties(self):
        buffer = BytesIO()
        Image.new("RGB", (640, 360), (230, 40, 40)).save(buffer, format="PNG")

        preview = caption_uploaded_image(buffer.getvalue())

        self.assertIn("landscape", preview.english.lower())
        self.assertIn("640x360", preview.english)
        self.assertIn("가로형", preview.korean)
        self.assertIn("640x360", preview.korean)


if __name__ == "__main__":
    unittest.main()
