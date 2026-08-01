# AI Lab Showcase

AI/Transformer 실습 과제 중 Lab 1, Lab 6, Lab 9만 Streamlit으로 정리한 제출용 포트폴리오 대시보드입니다.

Streamlit 앱에서는 무거운 모델을 시작 시 다운로드하지 않고, 저장된 데이터와 이미지 메타데이터를 사용해 빠르게 실행됩니다. 전체 모델 실행 흐름은 각 실습 노트북에서 이어서 설명할 수 있도록 분리했습니다.

## 포함된 실습

- Lab 1: 실제 영화명을 기준으로 한 리뷰 분석
- Lab 6: 실제 의상 이미지 기반 패션 검색
- Lab 9: 이미지 캡셔닝 데모

Lab 2와 Seq2Seq 관련 UI는 Streamlit 앱에서 제외했습니다.

## 실행 방법

```text
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Streamlit Community Cloud 설정

```text
Repository: pastar20250101-creator/practice_transformer4
Branch: main
Main file path: streamlit_app.py
```

## 파일 구조

```text
streamlit_app.py       # Streamlit UI
app_utils.py           # 데이터 로딩, 검색, 분석 함수
requirements.txt       # 배포 의존성
tests/test_app_utils.py # 핵심 로직 테스트
data/movie_reviews.csv # 영화 리뷰 분석용 로컬 CSV
```

## Lab 1: 영화 리뷰 분석

앱은 `data/movie_reviews.csv`를 읽어 영화별 리뷰 지표를 보여줍니다.

- 영화 선택 필터
- 리뷰 수, 평균 평점, 긍정/부정 비율
- 키워드 TOP 10
- 긍정 리뷰 키워드와 부정 리뷰 키워드 비교
- 리뷰 원문 일부 테이블

공개 영화 리뷰 데이터셋 구조를 참고하되, Streamlit 실행 중 실시간 무단 크롤링은 하지 않습니다. 앱 안에도 NSMC, Mendeley Korean Movie Reviews 같은 참고 출처를 명시했습니다.

## Lab 6: 패션 검색

로컬 과제 이미지와 공개 이미지 URL을 함께 사용해 20개 이상의 패션 이미지 보드를 구성했습니다.

- 카테고리: streetwear, minimal, office, casual, dress, sneakers, outer, accessories
- 검색 예시: `black street sneaker`, `minimal office`, `dress`, `bag`
- 이미지 카드: 스타일명, 카테고리, 태그, 설명, 출처 표시
- 배포 앱에서는 CLIP 모델 대신 태그 기반 검색으로 유사도 검색 개념을 시연합니다.

실제 CLIP 흐름은 이미지와 텍스트를 벡터로 변환한 뒤 cosine similarity로 가까운 항목을 찾는 방식입니다.

## Lab 9: 이미지 캡셔닝

기존 COCO 이미지와 사용자 이미지를 선택해 캡션 preview를 확인합니다.

- 이미지 선택
- 영어 캡션 preview
- 한국어 설명 preview
- BLIP -> BLEU -> Translation 파이프라인 표

Streamlit 앱에서는 모델 다운로드 없이 저장된 preview 로직을 사용합니다. 실제 모델 실행은 노트북에서 BLIP, BLEU 평가, 번역 단계로 확장할 수 있습니다.

## 검증

```text
python -m unittest tests.test_app_utils -v
python -m py_compile app_utils.py streamlit_app.py
```

## 보안

API 키, 비밀키, 토큰은 코드에 넣지 않습니다. 외부 API를 붙이는 경우 로컬에서는 `.env`, Streamlit Community Cloud에서는 `Secrets`를 사용합니다.
