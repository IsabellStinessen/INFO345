# Recommender Systems - Mood-Aware Hybrid Music Recommender
Group project developed as part of **INFO345: Research Topics in Recommender Systems** at the University of Bergen (Autumn 2025).

The project implements a **hybrid music recommendation system** that combines content-based filtering, collaborative filtering, and mood-based scoring to recommend songs based on both a user's listening history and their current emotional state.

The project was implemented in Python using Jupyter Notebook.

## Technologies
* Python
* Jupyter Notebook
* scikit-learn
* Pandas
* NumPy

## Data
* **[Million Song Dataset](http://millionsongdataset.com/)** - ~50,700 tracks with audio features such as valence, energy, danceability, tempo and mode
* **[Echo Nest Taste Profile Subset](http://millionsongdataset.com/tasteprofile/)** - ~9.7 million real user listening records (`user_id`, `track_id`, `playcount`)

Both datasets share a `track_id`, allowing audio features to be joined with real listening behaviour.

## Approach
* Cleaned and preprocessed both datasets, handling missing values, duplicates and redundant columns
* Encoded categorical variables using multi-hot encoding (tags) and label encoding (artists)
* Normalized numerical audio features and explored feature correlations
* Built a rule-based **mood classification system**, mapping audio features onto four emotional quadrants (valence x arousal) based on the circumplex model of emotion
* Implemented content-based filtering using k-Nearest Neighbors over audio features
* Implemented collaborative filtering using Truncated SVD on a sparse user-item playcount matrix
* Combined CF, CB and mood scores into a weighted hybrid recommendation score
* Evaluated the models using Precision@K, Recall@K, Hit Rate@K, NDCG@K and MAP@K

## Results

The four recommendation approaches were compared at k=5:

| **Model** | **Precision@5** | **Recall@5** | **Hit Rate@5** | **NDCG@5** | **MAP@5** |
|-------|----------|----------|----------|----------|----------|
| Collaborative Filtering (CF) | **0.0568** | **0.0195** | **0.2264** | **0.1409** | **0.0100** |
| Hybrid (CF 0.7, CB 0.3, Mood 0.0) | 0.0360 | 0.0123 | 0.1510 | 0.1045 | 0.0074 |
| Hybrid (CF 0.4, CB 0.2, Mood 0.4) | 0.0321 | 0.0111 | 0.1344 | 0.0899 | 0.0063 |
| Hybrid (CF 0.2, CB 0.2, Mood 0.6) | 0.0258 | 0.0088 | 0.1125 | 0.0754 | 0.0050 |
| Content-Based Filtering (CB) | 0.0003 | 0.0001 | 0.0015 | 0.0009 | 0.0001 |

Collaborative filtering performed best across all ranking metrics, indicating it is most successful at predicting songs a user has already listened to. Content-based filtering performed weakly on its own but strengthened the hybrid scores when combined with CF. Hybrid models weighted more heavily toward mood scored lower on these metrics, since standard ranking metrics only measure whether a user has listened to a track before, and the mood labels used were not validated against real user feedback. This means the true value of the mood component is not fully captured by the metrics above.

Both traditional models and the hybrid model were also compared against random and popularity baselines, confirming that all models meaningfully outperformed random recommendations.

## Project contribution
I was primarily responsible for the **data preprocessing** and **results analysis** portions of the project, including cleaning and preparing both datasets, building the mood classification system, and interpreting the evaluation metrics and model comparisons. The project was completed in collaboration with two other group members.

## Repository contents
* `PreprocessingData.ipynb`- Jupyter Notebook containing the data cleaning, preprocessing, feature encoding/normalization and mood classification for both the item (music) and user-item (listening history) datasets
* `EvaluationPart1.ipynb` - Jupyter Notebook that splits the data into train/test sets and computes Precision@K, Recall@K, Hit Rate@K, NDCG@K and MAP@K for the hybrid model against the CF-only and CB-only baselines
* `EvaluationPart2.ipynb` - Jupyter Notebook comparing the hybrid model against random and popularity baselines using hit rate
* `INFO345.pdf` - PDF report describing the project in full, including methodology, figures and references
* `README.md` - Project overview and documentation
