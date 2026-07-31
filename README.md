# Transformer Practice - Lab 1, Lab 6, and Lab 9 Submission

AI Human Transformer practice assignment submission for Lab 1, Lab 6, and Lab 9.

## Included Labs

- Lab 1: Korean movie review analysis and visualization
- Lab 6: CLIP multimodal fashion image and text search
- Lab 9: Image captioning with BLIP and Korean translation

## Main Notebooks

Open and run:

```text
lab1_korean_movie_review_analysis/lab1_korean_movie_review_analysis.ipynb
lab6_clip_multimodal_fashion_search/lab6_clip_multimodal_fashion_search.ipynb
lab9_image_captioning/lab9_image_captioning.ipynb
```

## What Was Completed

### Lab 1

- Loaded and cleaned Korean movie review text.
- Tokenized Korean text with POS filtering and stopword removal.
- Compared basic and improved keyword extraction.
- Saved analysis outputs such as top-token text files, bar plots, word clouds, and POS comparison charts.

### Lab 6

- Loaded fashion image data.
- Encoded text queries with CLIP text embeddings.
- Encoded fashion images with CLIP image embeddings.
- Implemented image-to-image search by cosine similarity.
- Implemented text-to-image fashion style search.

### Lab 9

- Generated image captions with BLIP.
- Compared conditional and unconditional caption generation.
- Evaluated captions with BLEU score.
- Translated generated captions into Korean.

## Output Files

Lab 1 generated files are in:

```text
lab1_korean_movie_review_analysis/output/
```

Lab 6 image assets are in:

```text
lab6_clip_multimodal_fashion_search/fashion_images/
```

Lab 9 image assets and caption data are in:

```text
lab9_image_captioning/
```

The large MSCOCO Korean caption JSON is tracked with Git LFS.

## Environment Variables

Lab 6 reads the Unsplash API key from an environment variable instead of storing it in the notebook.

Use `.env.example` as the template:

```text
UNSPLASH_API_KEY=your_unsplash_access_key_here
```
