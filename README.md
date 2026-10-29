# Restaurant Data – Machine Learning Project (Cognifyz Internship)

Four machine-learning / data-analysis tasks on the Zomato restaurant dataset (9,551 restaurants, 21 columns, 15 countries):

| # | Task | Technique | Script |
|---|------|-----------|--------|
| 1 | Predict restaurant ratings | Regression (Linear Regression, Decision Tree, Random Forest) | `task1_rating_prediction.py` |
| 2 | Restaurant recommendation | Content-based filtering (TF-IDF + cosine similarity) | `task2_restaurant_recommendation.py` |
| 3 | Cuisine classification | Classification (Logistic Regression, Random Forest) | `task3_cuisine_classification.py` |
| 4 | Location-based analysis | Geographic visualisation and group statistics | `task4_location_analysis.py` |

---

## 1. Project structure

```
cognifyz Internship/
├── Dataset_.csv                          <- the dataset (see "Dataset" below)
├── requirements.txt
├── README.md
├── task1_rating_prediction.py
├── task2_restaurant_recommendation.py
├── task3_cuisine_classification.py
├── task4_location_analysis.py
└── outputs/                              <- created automatically when you run the scripts
```

## 2. Setup

Requires **Python 3.9 or newer** (developed and tested on Python 3.12).

```bash
pip install -r requirements.txt
```

Libraries used: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `folium`.

## 3. How to run

Put the dataset CSV in the same folder as the scripts (any file whose name starts with `Dataset`, e.g. `Dataset_.csv`, is detected automatically). Then run each script from that folder:

```bash
python task1_rating_prediction.py
python task2_restaurant_recommendation.py
python task3_cuisine_classification.py
python task4_location_analysis.py
```

- Each script is **independent** – you can run them in any order, or just one.
- No chart windows pop up. Everything is saved to the `outputs/` folder.
- Tasks 1 and 3 train several models and may take 1–3 minutes.

## 4. What gets saved in `outputs/`

Every script writes a `..._results.txt` file containing everything it printed, plus charts (`.png`) and tables (`.csv`).

| Task | Files |
|------|-------|
| **1** | `task1_rating_prediction_results.txt`, `task1_model_metrics.csv`, `task1_feature_importance_B.csv`, `task1_feature_importance_C.csv`, `task1_00_rating_price_distribution.png`, `task1_01_actual_vs_predicted.png`, `task1_02_feature_importance.png` |
| **2** | `task2_restaurant_recommendation_results.txt`, `task2_user_1..5_recommendations.csv`, `task2_similar_restaurants_example.csv`, `task2_evaluation_metrics.csv`, `task2_00_rating_price_distribution.png` |
| **3** | `task3_cuisine_classification_results.txt`, `task3_model_comparison.csv`, `task3_per_cuisine_report.csv`, `task3_accuracy_india_vs_outside.csv`, `task3_top_confusions.csv`, `task3_01_confusion_matrix_and_f1.png`, `task3_00_rating_price_distribution.png` |
| **4** | `task4_location_analysis_results.txt`, `restaurant_map.html` (interactive map – open in a browser), `task4_country_summary.csv`, `task4_city_statistics.csv`, `task4_locality_statistics.csv`, `task4_region_comparison.csv`, `task4_01_static_maps.png`, `task4_02_top_cities_and_localities.png`, `task4_03_city_rating_and_price_ranking.png`, `task4_04_cuisine_heatmap.png`, `task4_00_rating_price_distribution.png` |

---

## 5. Dataset notes and data-quality decisions

- **Unrated restaurants:** 2,148 restaurants (22%) have `Aggregate rating = 0`, meaning "not rated". Average ratings in Task 4 use rated restaurants only.
- **Data leakage avoided:** `Rating color` and `Rating text` are derived directly from `Aggregate rating`, so they are **not** used as features in Task 1.
- **Currencies:** `Average Cost for two` is in 12 different currencies. It is converted to `cost_vs_country` (cost ÷ the country's median cost) so it is comparable across countries.
- **Missing values:** 9 missing `Cuisines` values are filled with `"Unknown"` (excluded from Tasks 2 and 3).
- **Invalid coordinates:** about 5% of rows have a latitude or longitude of 0 (missing location). They are excluded from the maps only.
- **Encoding:** some city names (e.g. "Brasília", "São Paulo") are already corrupted in the source CSV. This does not affect results.
- **Country names** come from the standard Zomato country-code table; the CSV itself only contains numeric codes.

---

## 6. Methods and results

### Task 1 – Predict restaurant ratings
- **Preprocessing:** one-hot encoding for categorical columns (inside the pipeline, so fitted on training data only), scaling for numeric columns, 80/20 train/test split.
- **Models:** Linear Regression, Decision Tree (depth tuned with 5-fold cross-validation), Random Forest.
- **Three scenarios** are reported so the numbers are interpreted honestly:

| Scenario | Best model | RMSE | MAE | R² |
|---|---|---|---|---|
| A: all restaurants | Random Forest | 0.295 | 0.193 | 0.962 |
| B: rated restaurants only | Random Forest | 0.326 | 0.240 | 0.657 |
| C: rated only, without `Votes` | Random Forest | 0.377 | 0.290 | 0.541 |

- Scenario A's high R² is partly because unrated restaurants (rating 0) are easy to spot. **Scenario B is the realistic result** (about 0.24 stars of average error). A chain-aware split gives R² = 0.625, so the effect of the same chain appearing in train and test is small.
- **Most influential features:** `Votes` by far, then city/location and cuisine. Without votes, location dominates.

### Task 2 – Restaurant recommendation
- Cuisines are converted to TF-IDF vectors; a user's preferred cuisines are compared with each restaurant by cosine similarity.
- Score = 0.60 × cuisine similarity + 0.25 × price match + 0.15 × rating. Filters: city, minimum votes, optional table booking / online delivery. Each chain is shown once.
- `recommend(cuisines, price_range, city, ...)` is the main function; `recommend_similar(name)` finds restaurants similar to one the user likes.
- **Evaluation (299 simulated users, top-10):** 93% of recommendations contain the preferred cuisine, 99.5% are within ±1 price tier, ratings are 0.28 stars above the city average, and 23% of the catalogue is ever recommended. There is no real user history in the data, so these are proxy metrics.

### Task 3 – Cuisine classification
- **Label:** the first-listed (primary) cuisine, keeping the 10 most common classes. The `Cuisines` column is never used as a feature.
- **Split:** chain-aware (no restaurant name appears in both train and test).

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Majority baseline | 0.435 | 0.061 |
| Random Forest (no name) | 0.373 | 0.291 |
| Logistic Regression + name words | 0.570 | **0.514** |
| Random Forest + name words | **0.617** | 0.499 |

- Location, price and ratings alone are weak predictors; **words in the restaurant name** are the key signal. Macro-F1 is the main metric because accuracy alone rewards always guessing the largest class.
- **Challenges / biases:** class imbalance (North Indian is 43% of the data), geographic bias (about 97% of test rows are in India), and label ambiguity (a restaurant listing several cuisines has an arbitrary "primary" one). Hardest classes: Continental and American.

### Task 4 – Location-based analysis
- Static maps, an interactive map (`restaurant_map.html`), concentration by country / city / locality, and statistics by city (ratings, price range, cuisines).
- **Key findings:** about 91% of restaurants are in India and **83% are in Delhi-NCR** (New Delhi alone ≈ 57%). Gurgaon is the premium NCR city (highest price range and table-booking share). Premium localities include Gurgaon Sector 29, Cyber Hub, Hauz Khas Village and Khan Market.
- **Caution:** non-NCR cities contribute only about 20 restaurants each, almost all rated, so they look like a "top-listed" sample. Their higher average ratings do **not** mean those cities have better food.

---

## 7. Limitations and possible improvements

- Ratings and votes come from a single snapshot; votes are partly a *result* of popularity, not a cause of quality.
- Task 2 has no real user interaction data, so it cannot be evaluated against true preferences or personalised.
- Task 3 could be framed as multi-label classification (predict all cuisines) instead of using only the primary cuisine.
- With more data outside Delhi-NCR, the city comparisons in Task 4 would be far more reliable.

## 8. Reproducibility

All random operations use a fixed seed (`RANDOM_STATE = 42`), so results are reproducible.
