"""
task3_cuisine_classification.py
Run with:  python task3_cuisine_classification.py
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
plt.tight_layout(); save_fig("task3_00_rating_price_distribution")

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

# ===== task3_cuisine_classification.py =====
# ---
# # Task 3 – Cuisine Classification
#
# **Objective:** classify a restaurant's cuisine from its other attributes.
#
# A restaurant can list several cuisines (e.g. "North Indian, Chinese"), so I use the **first-listed (primary) cuisine** as the label and keep the **10 most common** primary cuisines. Features: location (country, city, lat/long), price range, relative cost, rating, votes, and the service flags. The `Cuisines` column is never used as a feature (it *is* the label). I compare models **without** and **with** the restaurant name (TF-IDF of its words).
#
# Models: Logistic Regression and Random Forest (both class-weighted because the classes are imbalanced), compared against a majority-class baseline. The train/test split is **chain-aware** (no restaurant name appears in both sets) so the model can't just memorise branches of the same chain.

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

t3 = df[df["Cuisines"] != "Unknown"].copy()
top10 = t3["primary_cuisine"].value_counts().head(10).index.tolist()
t3 = t3[t3["primary_cuisine"].isin(top10)].reset_index(drop=True)
print("Classes kept:", top10, "| rows:", len(t3))
print((t3["primary_cuisine"].value_counts(normalize=True) * 100).round(1).to_string())

NUM3 = ["Longitude", "Latitude", "Price range", "cost_vs_country", "log_votes", "Aggregate rating"]
CAT3 = ["Country Code", "City"]

def make_pre3(use_name):
    parts = [("num", StandardScaler(), NUM3),
             ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=30), CAT3),
             ("bin", "passthrough", BIN_FEATS)]
    if use_name:   # words in the restaurant name ("pizza", "bakery", "sweets", "cafe"...) are strong cuisine hints
        parts.append(("name", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "Restaurant Name"))
    return ColumnTransformer(parts)

# Chain-aware split: outlets of the same chain share a name AND a cuisine, so they must not be split across train/test.
X3, y3 = t3[NUM3 + CAT3 + BIN_FEATS + ["Restaurant Name"]], t3["primary_cuisine"]
tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE).split(X3, groups=t3["Restaurant Name"]))
Xtr3, Xte3, ytr3, yte3 = X3.iloc[tr_i], X3.iloc[te_i], y3.iloc[tr_i], y3.iloc[te_i]
print(f"\nTrain: {len(Xtr3)} | Test: {len(Xte3)} | majority-class share in test: {(yte3 == 'North Indian').mean():.1%}")

experiments = {   # name -> (classifier, use_name_feature)
  "Majority baseline":              (DummyClassifier(strategy="most_frequent"), False),
  "Logistic Regression":            (LogisticRegression(max_iter=3000, class_weight="balanced"), False),
  "Random Forest":                  (RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE), False),
  "Logistic Regression + name":     (LogisticRegression(max_iter=3000, class_weight="balanced", C=5), True),
  "Random Forest + name":           (RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE), True),
}
rows, fitted3 = [], {}
for name, (c, use_name) in experiments.items():
    p = Pipeline([("pre", make_pre3(use_name)), ("model", c)]).fit(Xtr3, ytr3)
    pred = p.predict(Xte3)
    pr, rc, f1, _ = precision_recall_fscore_support(yte3, pred, average="macro", zero_division=0)
    rows.append({"Model": name, "Accuracy": accuracy_score(yte3, pred), "Macro precision": pr, "Macro recall": rc,
                 "Macro F1": f1, "Weighted F1": precision_recall_fscore_support(yte3, pred, average="weighted", zero_division=0)[2]})
    fitted3[name] = (p, pred)
res3 = pd.DataFrame(rows)
print(res3.round(3).to_string(index=False))
save_table(res3.round(3), "task3_model_comparison", index=False)

best3 = res3.iloc[1:].sort_values("Macro F1").iloc[-1]["Model"]   # best by macro-F1 (fair to rare cuisines)
pred3 = fitted3[best3][1]
print("Best model by macro-F1:", best3, "\n")
print(classification_report(yte3, pred3, zero_division=0))

labels = sorted(top10)
cm = confusion_matrix(yte3, pred3, labels=labels, normalize="true")
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw={"width_ratios": [1.25, 1]})
sns.heatmap(cm, annot=True, fmt=".2f", cmap="Reds", xticklabels=labels, yticklabels=labels, ax=ax[0], cbar=False)
ax[0].set_title(f"{best3}: confusion matrix (rows = true cuisine, normalised)"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True")
plt.setp(ax[0].get_xticklabels(), rotation=45, ha="right")
f1s = pd.Series(precision_recall_fscore_support(yte3, pred3, labels=labels, zero_division=0)[2], index=labels).sort_values()
f1s.plot.barh(ax=ax[1], color="#3b82f6"); ax[1].set_title("F1 per cuisine"); ax[1].set_xlim(0, 1)
plt.tight_layout(); save_fig("task3_01_confusion_matrix_and_f1")

# ### Challenges & biases – performance across cuisines and regions

# 1) Class imbalance: does the model do worse on rare cuisines?
rep = pd.DataFrame(classification_report(yte3, pred3, output_dict=True, zero_division=0)).T.loc[labels, ["precision", "recall", "f1-score", "support"]]
display(rep.sort_values("support", ascending=False).round(3))
save_table(rep.sort_values("support", ascending=False).round(3), "task3_per_cuisine_report")

# 2) Geographic bias: accuracy for India vs. rest of the world
is_india = Xte3["Country Code"] == "1"
geo_acc = pd.Series({
    "India (%d test rows)" % is_india.sum():         accuracy_score(yte3[is_india],  pred3[is_india]),
    "Outside India (%d test rows)" % (~is_india).sum(): accuracy_score(yte3[~is_india], pred3[~is_india]),
})
display(geo_acc.round(3).to_frame("accuracy"))
save_table(geo_acc.round(3).to_frame("accuracy"), "task3_accuracy_india_vs_outside")

# 3) Which cuisines get confused with each other most often?
conf = pd.crosstab(yte3.values, pred3, rownames=["true"], colnames=["pred"])
pairs = conf.stack().reset_index(name="n"); pairs = pairs[pairs["true"] != pairs["pred"]].sort_values("n", ascending=False).head(6)
print("Most frequent confusions:"); display(pairs.reset_index(drop=True))
save_table(pairs.reset_index(drop=True), "task3_top_confusions", index=False)

# ### Task 3 – interpretation
# | Model | Accuracy | Macro-F1 |
# |---|---|---|
# | Majority baseline ("North Indian") | 0.435 | 0.061 |
# | Random Forest (no name) | 0.373 | 0.291 |
# | Logistic Regression + name | 0.570 | **0.514** |
# | Random Forest + name | **0.617** | 0.499 |
#
# - **Location, price and ratings barely identify a cuisine.** Without the name, even the best model is below the majority baseline on *accuracy* (it trades accuracy for finding rare cuisines, hence the much better macro-F1). **Name words are the key signal** ("bakery", "sweets", "pizza", "cafe"...). I report macro-F1 as the main metric because accuracy alone rewards always guessing the biggest class.
# - **Strongest classes:** North Indian (F1 0.66), Bakery (0.65), Mithai (0.63). **Weakest:** Continental (0.25 – many false alarms) and American (0.37 – 66% of true American restaurants are predicted as North Indian).
# - **Challenges / biases**
#   - *Class imbalance:* North Indian is 43% of the data, so it dominates predictions and absorbs mistakes of other classes (most frequent errors: North Indian ↔ Continental / Chinese / Fast Food).
#   - *Geographic bias:* ~97% of the test rows are in India, so the model effectively learns the Indian market. The higher accuracy outside India (71%) comes from only 55 test rows and is not reliable.
#   - *Label ambiguity:* a restaurant listing "Cafe, Italian, Desserts" has an arbitrary "primary" cuisine, so some "errors" are really valid alternative labels. Treating this as a multi-label problem (predict all cuisines) would be a natural next step.

print("\nDone. All outputs were saved in:", OUT)
