"""
task2_restaurant_recommendation.py
Run with:  python task2_restaurant_recommendation.py
Expects the dataset (Dataset_.csv) in the same folder.
All charts / tables / printed results are saved in an 'outputs' folder next to this script.
"""

# ===== Shared setup & preprocessing =====
import os, sys, glob
import matplotlib
matplotlib.use("Agg")                         # charts are saved as PNG files instead of opening windows
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")           # every result is saved in this folder
os.makedirs(OUT, exist_ok=True)
SCRIPT = os.path.splitext(os.path.basename(__file__))[0]

class _Tee:                                   # prints to the terminal AND to a text file
    def __init__(self, *streams): self.streams = streams
    def write(self, text):
        for s in self.streams: s.write(text)
    def flush(self):
        for s in self.streams: s.flush()
try: sys.stdout.reconfigure(encoding="utf-8")
except Exception: pass
_log = open(os.path.join(OUT, f"{SCRIPT}_results.txt"), "w", encoding="utf-8")
sys.stdout = _Tee(sys.__stdout__, _log)
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
def display(obj):
    print(obj.to_string() if hasattr(obj, 'to_string') else obj)

def save_fig(name):
    path = os.path.join(OUT, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close("all")
    print(f"[saved chart] {path}")

def save_table(obj, name, index=True):
    path = os.path.join(OUT, f"{name}.csv")
    obj.to_csv(path, index=index, encoding="utf-8-sig")
    print(f"[saved table] {path}")

warnings.filterwarnings("ignore")
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GroupShuffleSplit

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 110
RANDOM_STATE = 42

_found = sorted(set(glob.glob(os.path.join(HERE, "Dataset*.csv")) + glob.glob(os.path.join(HERE, "dataset*.csv"))))
if not _found:
    raise SystemExit("Dataset not found. Put Dataset_.csv in the same folder as this script: " + HERE)
DATA = _found[0]
df = pd.read_csv(DATA)
df.columns = df.columns.str.strip()
print("Shape:", df.shape)
display(df.head(3))

# ## 0. Data understanding & shared preprocessing

print("Duplicate Restaurant IDs:", df["Restaurant ID"].duplicated().sum())
print("\nMissing values:")
print(df.isna().sum()[df.isna().sum() > 0])

print("\nRestaurants with rating 0 (unrated):", (df["Aggregate rating"] == 0).sum(),
      f"({(df['Aggregate rating'] == 0).mean():.1%})")
unr = df[df["Aggregate rating"] == 0]
print(f"Unrated restaurants: median votes = {unr['Votes'].median():.0f}; {(unr['Votes'] == 0).mean():.0%} have exactly 0 votes. "
      f"Rated restaurants: median votes = {df.loc[df['Aggregate rating'] > 0, 'Votes'].median():.0f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
sns.histplot(df["Aggregate rating"], bins=30, ax=ax[0], color="#e23744")
ax[0].set_title("Distribution of Aggregate rating (note the spike at 0)")
sns.countplot(x="Price range", data=df, ax=ax[1], color="#3b82f6")
ax[1].set_title("Price range")
plt.tight_layout(); save_fig("task2_00_rating_price_distribution")

# ---- Shared cleaning / feature engineering ----
df["Cuisines"] = df["Cuisines"].fillna("Unknown")            # only 9 missing values
df["primary_cuisine"] = df["Cuisines"].str.split(",").str[0].str.strip()
df["n_cuisines"] = df["Cuisines"].apply(lambda s: 0 if s == "Unknown" else len(s.split(",")))

# Binary Yes/No columns -> 1/0
BIN_COLS = ["Has Table booking", "Has Online delivery", "Is delivering now", "Switch to order menu"]
for c in BIN_COLS:
    df[c + "_bin"] = (df[c] == "Yes").astype(int)
BIN_FEATS = [c + "_bin" for c in BIN_COLS]
print({c: df[c].value_counts().to_dict() for c in BIN_COLS})

# Costs are in different currencies -> express cost relative to the country median
df["cost_vs_country"] = df["Average Cost for two"] / df.groupby("Country Code")["Average Cost for two"].transform("median")
df["log_votes"] = np.log1p(df["Votes"])

# Country names (Zomato country-code table) - sanity check using the most common city per code
COUNTRY = {1:"India",14:"Australia",30:"Brazil",37:"Canada",94:"Indonesia",148:"New Zealand",162:"Philippines",
           166:"Qatar",184:"Singapore",189:"South Africa",191:"Sri Lanka",208:"Turkey",214:"UAE",215:"United Kingdom",216:"United States"}
df["Country"] = df["Country Code"].map(COUNTRY)
display(df.groupby("Country")["City"].agg(lambda s: s.value_counts().index[0]).to_frame("most common city").T)

# Country Code is a category, not a number
df["Country Code"] = df["Country Code"].astype(str)

# ===== task2_restaurant_recommendation.py =====
# ---
# # Task 2 – Restaurant Recommendation System (content-based filtering)
#
# **Objective:** recommend restaurants that match a user's stated preferences.
#
# **Approach**
# - Each restaurant's cuisines are turned into a **TF-IDF vector** (rarer cuisines carry more weight than ubiquitous ones like *North Indian*).
# - The user's preferred cuisines are encoded in the same space and compared with **cosine similarity**.
# - Final score = `0.60 × cuisine similarity + 0.25 × price match + 0.15 × rating`.
# - A chain is shown once (its best-scoring outlet) so the list isn't 5 branches of the same place.
# - Hard filters: city, minimum votes (so a 5.0 from 2 votes isn't promoted), optional table-booking / online-delivery requirements. Unrated restaurants are excluded.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

rec_df = df[df["Cuisines"] != "Unknown"].reset_index(drop=True)
tfidf = TfidfVectorizer(tokenizer=lambda s: [c.strip().lower() for c in s.split(",")], token_pattern=None, lowercase=False)
cuisine_matrix = tfidf.fit_transform(rec_df["Cuisines"])
print("Restaurants:", cuisine_matrix.shape[0], "| distinct cuisines:", cuisine_matrix.shape[1])

W_CUISINE, W_PRICE, W_RATING = 0.60, 0.25, 0.15

def recommend(cuisines, price_range=None, city=None, top_n=5, min_votes=10,
              need_table_booking=False, need_online_delivery=False, exclude_id=None, one_per_chain=True):
    """Return the top_n restaurants for a user's preferences."""
    user_vec = tfidf.transform([", ".join(cuisines)])
    if user_vec.nnz == 0:
        raise ValueError(f"None of {cuisines} are known cuisines. Examples: {list(tfidf.get_feature_names_out()[:15])}")
    mask = (rec_df["Votes"] >= min_votes) & (rec_df["Aggregate rating"] > 0)
    if city:                 mask &= rec_df["City"].str.lower() == city.lower()
    if need_table_booking:   mask &= rec_df["Has Table booking_bin"] == 1
    if need_online_delivery: mask &= rec_df["Has Online delivery_bin"] == 1
    if exclude_id is not None: mask &= rec_df["Restaurant ID"] != exclude_id
    idx = np.where(mask)[0]
    if len(idx) == 0: return pd.DataFrame()
    sim = cosine_similarity(user_vec, cuisine_matrix[idx]).ravel()
    pool = rec_df.iloc[idx].copy()
    pool["cuisine_sim"] = sim
    pool["price_match"] = 1 - (pool["Price range"] - price_range).abs() / 3 if price_range else 1.0
    pool["score"] = W_CUISINE * pool["cuisine_sim"] + W_PRICE * pool["price_match"] + W_RATING * pool["Aggregate rating"] / 5
    pool = pool[pool["cuisine_sim"] > 0]                      # must share at least one cuisine
    cols = ["Restaurant Name", "City", "Locality", "Cuisines", "Price range", "Aggregate rating", "Votes", "cuisine_sim", "score"]
    pool = pool.sort_values("score", ascending=False)
    if one_per_chain:                                          # show a chain once (its best-scoring outlet)
        pool = pool.drop_duplicates("Restaurant Name")
    return pool.head(top_n)[cols].round(3).reset_index(drop=True)

def recommend_similar(name, top_n=5):
    """Item-to-item: restaurants most similar to a restaurant the user already likes."""
    i = rec_df.index[rec_df["Restaurant Name"].str.lower() == name.lower()][0]
    r = rec_df.loc[i]
    return recommend(r["Cuisines"].split(", "), r["Price range"], r["City"], top_n, exclude_id=r["Restaurant ID"])

# ### Testing with sample user preferences

tests = {
 "User 1 – Italian lover, mid-budget, New Delhi":   dict(cuisines=["Italian"], price_range=3, city="New Delhi"),
 "User 2 – Chinese + Thai, cheap, Gurgaon":         dict(cuisines=["Chinese", "Thai"], price_range=2, city="Gurgaon"),
 "User 3 – Cafe & Desserts, wants table booking":   dict(cuisines=["Cafe", "Desserts"], price_range=2, city="Mumbai", need_table_booking=True),
 "User 4 – Fine-dining Japanese/Sushi, expensive":  dict(cuisines=["Japanese", "Sushi"], price_range=4, city="New Delhi"),
 "User 5 – Pizza/Fast Food, cheap, wants delivery": dict(cuisines=["Pizza", "Fast Food"], price_range=1, city="New Delhi", need_online_delivery=True),
}
for title, kw in tests.items():
    print("\n" + title); _r = recommend(**kw); display(_r)
    save_table(_r, "task2_" + title.split(" – ")[0].replace(" ", "_").lower() + "_recommendations", index=False)

print("Item-to-item example: restaurants similar to one the user already likes")
sample_name = rec_df.loc[rec_df["Votes"].idxmax(), "Restaurant Name"]
print("Seed restaurant:", sample_name)
_sim = recommend_similar(sample_name); display(_sim)
save_table(_sim, "task2_similar_restaurants_example", index=False)

# ### Evaluating recommendation quality
# There are no real user–restaurant interactions in this dataset (no clicks, no individual reviews), so standard metrics like recall@k against true user history aren't possible. Instead I evaluate with **simulated users**: 300 random restaurants are used as "seeds" (their city, cuisines and price range become the user's preferences; the seed itself is excluded). I then measure, over the top-10 results:
# - **Cuisine match** – share of recommendations containing the user's *first-choice* cuisine
# - **Price fit** – share within ±1 price tier of the preference
# - **Rating lift** – mean rating of recommendations vs. the mean rating of the whole city pool
# - **Coverage** – share of the catalogue that ever gets recommended (low coverage = popularity bias)

rng = np.random.default_rng(RANDOM_STATE)
elig = rec_df[(rec_df["Votes"] >= 10) & (rec_df["Aggregate rating"] > 0)]
city_sizes = elig["City"].value_counts()
seeds = elig[elig["City"].isin(city_sizes[city_sizes >= 30].index)].sample(300, random_state=RANDOM_STATE)

K = 10; cm, pf, lift, covered, n_ok = [], [], [], set(), 0
for _, s in seeds.iterrows():
    prefs = s["Cuisines"].split(", ")
    recs = recommend(prefs, s["Price range"], s["City"], top_n=K, exclude_id=s["Restaurant ID"])
    if recs.empty: continue
    n_ok += 1
    cm.append(recs["Cuisines"].str.contains(prefs[0], regex=False).mean())
    pf.append(((recs["Price range"] - s["Price range"]).abs() <= 1).mean())
    pool_mean = elig.loc[elig["City"] == s["City"], "Aggregate rating"].mean()
    lift.append(recs["Aggregate rating"].mean() - pool_mean)
    covered.update(recs["Restaurant Name"] + recs["Locality"])

summary = pd.Series({
    "Simulated users evaluated": n_ok,
    f"Cuisine match@{K}": np.mean(cm),
    f"Price within ±1 tier@{K}": np.mean(pf),
    "Mean rating lift vs. city average": np.mean(lift),
    "Catalogue coverage": len(covered) / len(elig),
})
display(summary.to_frame("value").round(3))
save_table(summary.to_frame("value").round(3), "task2_evaluation_metrics")

# ### Task 2 – interpretation
# - The recommendations look sensible: Italian + New Delhi returns Italian-only restaurants at the requested price tier; the Japanese/Sushi user gets the Taj, Leela and Hyatt fine-dining places; multi-cuisine users get restaurants that cover *both* cuisines first (cuisine similarity = 1.0) before partial matches.
# - On 299 simulated users (top-10): **92.9% contain the preferred cuisine**, **99.5% are within ±1 price tier**, and the average rating of recommendations is **0.28 stars above the city average**. **Catalogue coverage is only ~23%**, meaning the ranker keeps favouring highly rated, well-voted places (popularity bias) – a lower rating weight or a diversity re-ranking would widen it.
# - **Limitations:** content-based filtering only knows cuisine, price, rating and a few flags, so it can't capture ambience or personal taste; there's no user history to personalise or to evaluate against, so these metrics are *proxies*; and cities with ~20 listings (most non-NCR cities) give very short candidate lists (strict filters such as delivery + cheap can leave only 1–2 results).

print("\nDone. All outputs were saved in:", OUT)
