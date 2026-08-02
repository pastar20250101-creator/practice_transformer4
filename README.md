# AI Lab Tools

Transformer 실습 1, 2, 6, 9를 Streamlit에서 바로 써볼 수 있는 도구형 앱으로 정리한 프로젝트입니다.

## 포함 기능

- Lab 1: 영화 리뷰 분석기
- Lab 2: 한국어 → 영어 번역기
- Lab 6: 패션 검색기
- Lab 9: 이미지 캡셔닝

## 앱 구성

### Lab 1 영화 리뷰 분석기

리뷰를 직접 입력하거나 영화 샘플을 선택하면 감성 점수, 긍정/부정 히트 수, 토큰 수, 핵심 키워드를 확인할 수 있습니다.

### Lab 2 번역기

실습 데이터의 한국어/영어 문장 쌍을 사용해 입력 문장과 가장 가까운 학습 문장을 찾고 영어 번역을 보여줍니다.

### Lab 6 패션 검색기

의상 이미지를 카테고리와 태그 기반으로 검색합니다. CLIP 실습의 핵심 아이디어인 텍스트-이미지 검색 흐름을 가볍게 시연합니다.

### Lab 9 이미지 캡셔닝

COCO 샘플 이미지와 한국어/영어 캡션을 확인하고, 사용자 업로드 이미지의 캡셔닝 프리뷰 화면도 볼 수 있습니다.

## 실행 방법

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Cloud 설정

```text
Repository: pastar20250101-creator/practice_transformer4
Branch: main
Main file path: streamlit_app.py
```

## 검증

```bash
python -m unittest tests.test_app_utils -v
python -m py_compile app_utils.py streamlit_app.py
```

## 배포 안정성

- Streamlit 시작 시 CLIP, BLIP 같은 무거운 모델을 다운로드하지 않습니다.
- API 키와 토큰은 코드에 넣지 않습니다.
- 저장된 실습 데이터와 로컬 이미지로 동작합니다.
