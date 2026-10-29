"""
task4_location_analysis.py
Run with:  python task4_location_analysis.py
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
plt.tight_layout(); save_fig("task4_00_rating_price_distribution")

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

# ===== task4_location_analysis.py =====
# ---
# # Task 4 – Location-based Analysis
#
# **Objective:** geographical analysis of the restaurants – distribution on a map, concentration by city/locality, and statistics (ratings, cuisines, price range) by area.
#
# Notes: ratings of 0 mean "not rated", so *average ratings are computed on rated restaurants only*. Cost is not compared across countries (different currencies); I use the 1–4 **price range** instead.

# ---- Coordinate sanity check ----
geo = df.copy()
geo["zero_coord"] = (geo["Latitude"] == 0) | (geo["Longitude"] == 0)       # a 0 in either coordinate = missing/invalid location
print("Restaurants with a 0 latitude or longitude:", geo["zero_coord"].sum(), f"({geo['zero_coord'].mean():.1%})")
print(geo.loc[geo["zero_coord"], "City"].value_counts().head(5).to_dict())
geo_ok = geo[~geo["zero_coord"]].copy()
print("Rows with usable coordinates:", len(geo_ok), "of", len(geo))

# ---- Static maps: world view and India zoom, coloured by rating ----
rated_geo = geo_ok[geo_ok["Aggregate rating"] > 0]
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
sc = ax[0].scatter(rated_geo["Longitude"], rated_geo["Latitude"], c=rated_geo["Aggregate rating"], cmap="RdYlGn", s=6, alpha=.6)
ax[0].set_title("All rated restaurants (world)"); ax[0].set_xlabel("Longitude"); ax[0].set_ylabel("Latitude")
ind = rated_geo[(rated_geo["Country"] == "India")]
ax[1].scatter(ind["Longitude"], ind["Latitude"], c=ind["Aggregate rating"], cmap="RdYlGn", s=6, alpha=.6)
ax[1].set_title("India"); ax[1].set_xlabel("Longitude"); ax[1].set_ylabel("Latitude")
fig.colorbar(sc, ax=ax, label="Aggregate rating", shrink=.85)
save_fig("task4_01_static_maps")

# ---- Interactive map (saved as restaurant_map.html - open it in any browser) ----
import folium
from folium.plugins import MarkerCluster

def color_for(r):
    return "gray" if r == 0 else "darkgreen" if r >= 4.0 else "green" if r >= 3.5 else "orange" if r >= 2.5 else "red"

m = folium.Map(location=[20, 40], zoom_start=2, tiles="CartoDB positron")
cluster = MarkerCluster(name="Restaurants").add_to(m)
for _, r in geo_ok.iterrows():
    folium.CircleMarker([r["Latitude"], r["Longitude"]], radius=5, color=color_for(r["Aggregate rating"]), fill=True, fill_opacity=.8,
        tooltip=f'{r["Restaurant Name"]} – {r["Aggregate rating"]} ★ | {r["Cuisines"]}').add_to(cluster)
m.save(os.path.join(OUT, "restaurant_map.html"))
print("Saved outputs/restaurant_map.html  (marker colours: dark green ≥4.0, green ≥3.5, orange ≥2.5, red <2.5, grey = unrated)")

# ### Concentration of restaurants by country, city and locality

by_country = geo.groupby("Country").agg(restaurants=("Restaurant ID", "count"), cities=("City", "nunique")).sort_values("restaurants", ascending=False)
by_country["share_%"] = (by_country["restaurants"] / len(geo) * 100).round(1)
display(by_country)
save_table(by_country, "task4_country_summary")

top_cities = geo["City"].value_counts()
print("Cities in dataset:", len(top_cities), "| top 5 cities hold", f'{top_cities.head(5).sum() / len(geo):.1%}', "of all restaurants")
NCR = ["New Delhi", "Gurgaon", "Noida", "Faridabad", "Ghaziabad"]
print("Top 4 cities:", top_cities.head(4).to_dict())
print(f"Delhi-NCR cities together: {geo['City'].isin(NCR).sum()} restaurants = {geo['City'].isin(NCR).mean():.1%} of the dataset")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
top_cities.head(15).sort_values().plot.barh(ax=ax[0], color="#e23744"); ax[0].set_title("Top 15 cities by number of restaurants")
loc_counts = geo.groupby(["Locality", "City"]).size().sort_values(ascending=False).head(15)
loc_counts.rename(lambda x: x, level=0).sort_values().plot.barh(ax=ax[1], color="#3b82f6"); ax[1].set_title("Top 15 localities (densest areas)")
ax[1].set_ylabel(""); plt.tight_layout(); save_fig("task4_02_top_cities_and_localities")

# ### Statistics by city: ratings, price range, cuisines

def top_cuisine(s):
    return s.str.split(",").explode().str.strip().value_counts().index[0]

city_stats = geo.groupby("City").agg(
    country=("Country", "first"),
    restaurants=("Restaurant ID", "count"),
    avg_rating_rated=("Aggregate rating", lambda s: s[s > 0].mean()),
    pct_unrated=("Aggregate rating", lambda s: (s == 0).mean() * 100),
    avg_price_range=("Price range", "mean"),
    pct_online_delivery=("Has Online delivery_bin", lambda s: s.mean() * 100),
    pct_table_booking=("Has Table booking_bin", lambda s: s.mean() * 100),
    top_cuisine=("Cuisines", top_cuisine),
).round(2)
big = city_stats[city_stats["restaurants"] >= 20].sort_values("restaurants", ascending=False)
print("Cities with at least 20 restaurants:", len(big))
display(big.head(15))
save_table(big, "task4_city_statistics")

fig, ax = plt.subplots(1, 3, figsize=(17, 6))
big.sort_values("avg_rating_rated").tail(12)["avg_rating_rated"].plot.barh(ax=ax[0], color="#16a34a"); ax[0].set_title("Highest avg rating (cities ≥ 20 restaurants)")
big.sort_values("avg_rating_rated").head(12)["avg_rating_rated"].plot.barh(ax=ax[1], color="#dc2626"); ax[1].set_title("Lowest avg rating")
big.sort_values("avg_price_range").tail(12)["avg_price_range"].plot.barh(ax=ax[2], color="#f59e0b"); ax[2].set_title("Highest avg price range (1–4)")
for a in ax: a.set_ylabel("")
plt.tight_layout(); save_fig("task4_03_city_rating_and_price_ranking")

# Cuisine mix of the biggest cities (share of restaurants serving each cuisine)
big_cities = big.head(8).index
exp = geo[geo["City"].isin(big_cities)].assign(cuisine=lambda d: d["Cuisines"].str.split(",")).explode("cuisine")
exp["cuisine"] = exp["cuisine"].str.strip()
top_c = exp["cuisine"].value_counts().head(10).index
share = (exp[exp["cuisine"].isin(top_c)].groupby(["City", "cuisine"]).size().unstack(fill_value=0)
         .div(geo[geo["City"].isin(big_cities)].groupby("City").size(), axis=0) * 100)
plt.figure(figsize=(11, 4.5)); sns.heatmap(share.loc[big_cities, top_c], annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "% of restaurants"})
plt.title("Cuisine popularity by city (% of the city's restaurants serving it)"); plt.ylabel(""); plt.xlabel(""); save_fig("task4_04_cuisine_heatmap")

# Do bigger markets rate differently? And locality-level price/rating
loc_stats = geo.groupby(["City", "Locality"]).agg(n=("Restaurant ID", "count"), avg_rating=("Aggregate rating", lambda s: s[s > 0].mean()),
                                                avg_price=("Price range", "mean")).query("n >= 15").round(2)
save_table(loc_stats, "task4_locality_statistics")
print("Best-rated localities (>=15 restaurants):"); display(loc_stats.sort_values("avg_rating", ascending=False).head(5))
print("Priciest localities (>=15 restaurants):");   display(loc_stats.sort_values("avg_price", ascending=False).head(5))

# ### Why city averages must be read with care: Delhi-NCR vs. everywhere else

geo["region"] = np.where(geo["City"].isin(NCR), "Delhi-NCR", np.where(geo["Country"] == "India", "Other Indian cities", "Outside India"))
reg = geo.groupby("region").agg(restaurants=("Restaurant ID", "count"), cities=("City", "nunique"),
        avg_restaurants_per_city=("City", lambda s: len(s) / s.nunique()),
        pct_unrated=("Aggregate rating", lambda s: (s == 0).mean() * 100),
        avg_rating_rated=("Aggregate rating", lambda s: s[s > 0].mean()),
        median_votes=("Votes", "median"), avg_price_range=("Price range", "mean")).round(2)
display(reg)
save_table(reg, "task4_region_comparison")

# ### Task 4 – key insights
# - **The data is extremely concentrated.** India holds ~91% of restaurants, and the five Delhi-NCR cities (New Delhi, Gurgaon, Noida, Faridabad, Ghaziabad) alone hold **83%**; New Delhi by itself has 5,473 (57%). The US is next at ~4.5%. Any "global" conclusion is really a Delhi-NCR conclusion.
# - **About 5% of rows have a 0 latitude or longitude** (mostly Delhi-NCR) – these are missing locations and were excluded from the maps.
# - **Inside Delhi-NCR, Gurgaon looks like the "premium" city**: highest average price range (1.86), the most table-booking (18%) and a high online-delivery share (38%), and the best average rating of the big four (3.33). Noida (36%) and Faridabad (40%) have the highest share of unrated restaurants. North Indian is the most common cuisine in every NCR city, while Chinese leads in several smaller cities (e.g. Bhubaneshwar, Guwahati).
# - **Premium localities** (price range ≥ 2.7, ≥15 restaurants): Gurgaon *Sector 29* and *Cyber Hub*, Delhi *Hauz Khas Village* and *Khan Market* – all with solid ratings of ~3.8–3.9.
# - **Caution – the "NCR rates lower" pattern is largely a sampling artefact.** NCR averages 3.28 stars with 27% unrated and a median of 19 votes, whereas every other city contributes only ~9–20 restaurants, almost all rated (median ~200 votes), averaging 3.9–4.1. Those smaller cities look like a "top-listed" sample, so their higher averages shouldn't be read as "those cities have better food".

# ---
# # Summary
# | Task | Method | Headline result |
# |---|---|---|
# | 1. Rating prediction | Linear Regression, Decision Tree, Random Forest | Random Forest best. R² = 0.96 on all data (inflated by unrated zeros), **0.66 on rated restaurants**, 0.54 without Votes. `Votes` dominates; then city/location and cuisine. |
# | 2. Recommendation | TF-IDF cuisine similarity + price + rating | On 299 simulated users: 93% of top-10 contain the preferred cuisine, 99.5% are within ±1 price tier, and recommendations are rated +0.28 above the city average. |
# | 3. Cuisine classification | Logistic Regression / Random Forest, with & without name | Location/price alone is weak (macro-F1 0.29). Adding name words gives **57–62% accuracy, macro-F1 ≈ 0.51** (baseline: 43.5% accuracy, macro-F1 0.06). |
# | 4. Location analysis | Maps + group-by statistics | 83% of data is Delhi-NCR; interactive map saved as `restaurant_map.html`. City comparisons are confounded by how restaurants were sampled. |

print("\nDone. All outputs were saved in:", OUT)
