"""
task1_rating_prediction.py
Run with:  python task1_rating_prediction.py
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
plt.tight_layout(); save_fig("task1_00_rating_price_distribution")

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

# ===== task1_rating_prediction.py =====
# ---
# # Task 1 – Predict Restaurant Ratings
#
# **Objective:** predict `Aggregate rating` from the other features.
#
# **Plan:** preprocess → train/test split (80/20) → train Linear Regression, Decision Tree (tuned with cross-validation) and Random Forest → evaluate with MSE / RMSE / MAE / R² → interpret feature importance.
#
# Categorical variables (`Country Code`, `City`, `primary_cuisine`) are one-hot encoded (rare categories are grouped automatically so the model isn't flooded with columns). The encoder lives *inside* the pipeline, so it is fitted on training data only.
#
# Because ~22% of restaurants are unrated (rating 0, and usually very few votes), I report three scenarios so the results are interpreted honestly:
# - **A:** all restaurants, all features (what the task literally asks for)
# - **B:** only rated restaurants (rating > 0)
# - **C:** rated restaurants, *without* `Votes` – to see what the *restaurant's own attributes* can predict

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

TARGET = "Aggregate rating"
df["Country Code"] = df["Country Code"].astype(str)

NUM_FULL = ["Longitude", "Latitude", "Price range", "cost_vs_country", "n_cuisines", "log_votes"]
CAT_FEATS = ["Country Code", "City", "primary_cuisine"]

def make_pre(num_cols):
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=25), CAT_FEATS),
        ("bin", "passthrough", BIN_FEATS),
    ])

def run_regression(data, num_cols, label):
    X = data[num_cols + CAT_FEATS + BIN_FEATS]; y = data[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": GridSearchCV(DecisionTreeRegressor(random_state=RANDOM_STATE),
                                      {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [5, 10, 20]}, cv=5, scoring="r2"),
        "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE),
    }
    rows, fitted = [], {}
    for name, m in models.items():
        pipe = Pipeline([("pre", clone(make_pre(num_cols))), ("model", m)]).fit(Xtr, ytr)
        pred = np.clip(pipe.predict(Xte), 0, 5)
        mse = mean_squared_error(yte, pred)
        rows.append({"Scenario": label, "Model": name, "MSE": mse, "RMSE": np.sqrt(mse),
                     "MAE": mean_absolute_error(yte, pred), "R2": r2_score(yte, pred)})
        fitted[name] = (pipe, pred)
    return pd.DataFrame(rows), fitted, (Xtr, Xte, ytr, yte)

rated = df[df[TARGET] > 0]
resA, fitA, splitA = run_regression(df,    NUM_FULL,                                  "A: all restaurants")
resB, fitB, splitB = run_regression(rated, NUM_FULL,                                  "B: rated only")
resC, fitC, splitC = run_regression(rated, [c for c in NUM_FULL if c != "log_votes"], "C: rated only, no Votes")

results = pd.concat([resA, resB, resC], ignore_index=True)
print(results.round(4).to_string())
save_table(results.round(4), "task1_model_metrics", index=False)
print("Best decision-tree params (A):", fitA["Decision Tree"][0].named_steps["model"].best_params_)

# Actual vs predicted for the best model in Scenario A, plus residuals
best = resA.sort_values("R2").iloc[-1]["Model"]
_, _, _, yte = splitA
pred = fitA[best][1]
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(yte, pred, s=8, alpha=.35, color="#e23744")
ax[0].plot([0, 5], [0, 5], "k--", lw=1); ax[0].set_xlabel("Actual rating"); ax[0].set_ylabel("Predicted rating")
ax[0].set_title(f"{best}: actual vs predicted (Scenario A)")
sns.histplot(yte - pred, bins=40, ax=ax[1], color="#3b82f6"); ax[1].set_title("Residuals (actual - predicted)")
plt.tight_layout(); save_fig("task1_01_actual_vs_predicted")

# Which features matter most?  Permutation importance (on the TEST set, on the original columns)
def perm_imp(fitted, split, title, ax):
    pipe = fitted["Random Forest"][0]; _, Xte, _, yte = split
    r = permutation_importance(pipe, Xte, yte, n_repeats=5, random_state=RANDOM_STATE, scoring="r2", n_jobs=-1)
    imp = pd.Series(r.importances_mean, index=Xte.columns).sort_values()
    imp.plot.barh(ax=ax, color="#e23744"); ax.set_title(title); ax.set_xlabel("Drop in R² when shuffled")
    return imp.sort_values(ascending=False)

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
impB = perm_imp(fitB, splitB, "Random Forest – rated only (B)", ax[0])
impC = perm_imp(fitC, splitC, "Random Forest – rated only, no Votes (C)", ax[1])
plt.tight_layout(); save_fig("task1_02_feature_importance")
print("Top features (B):", list(impB.head(4).round(3).items()))
print("Top features (C):", list(impC.head(4).round(3).items()))
save_table(impB.round(4).to_frame("importance_B_rated_only"), "task1_feature_importance_B")
save_table(impC.round(4).to_frame("importance_C_no_votes"), "task1_feature_importance_C")

# Robustness check: the same chain (e.g. several outlets of one brand) can land in both train and test.
# Re-evaluate Scenario B's Random Forest with a chain-aware split (no restaurant name appears on both sides).
from sklearn.model_selection import GroupShuffleSplit
X_b = rated[NUM_FULL + CAT_FEATS + BIN_FEATS]; y_b = rated[TARGET]
tr_i, te_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE).split(X_b, groups=rated["Restaurant Name"]))
rf = Pipeline([("pre", clone(make_pre(NUM_FULL))), ("model", RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE))])
rf.fit(X_b.iloc[tr_i], y_b.iloc[tr_i])
print("Random Forest R² (random split, Scenario B):    %.3f" % resB.query("Model == 'Random Forest'")["R2"].iloc[0])
print("Random Forest R² (chain-aware split):           %.3f" % r2_score(y_b.iloc[te_i], rf.predict(X_b.iloc[te_i])))

# ### Task 1 – interpretation
# **Metrics (test set)**
#
# | Scenario | Best model | RMSE | MAE | R² |
# |---|---|---|---|---|
# | A: all restaurants | Random Forest | 0.295 | 0.193 | 0.962 |
# | B: rated only | Random Forest | 0.326 | 0.240 | 0.657 |
# | C: rated only, no Votes | Random Forest | 0.377 | 0.290 | 0.541 |
#
# - **Random Forest is best in every scenario**; Linear Regression is far behind in A (R² 0.745) because the "rating = 0 if unrated" spike is non-linear, and the tuned Decision Tree trails the forest.
# - **Scenario A's R² of 0.96 is flattering.** 22% of the targets are exactly 0 and those restaurants usually have almost no votes, so the model gets a lot of "credit" for spotting them. The realistic accuracy is Scenario B: predictions are off by about **0.24 stars on average** (MAE) and the model explains about two-thirds of the variance. A chain-aware split gives R² = 0.625, so branch-leakage is a minor effect.
# - **Most influential features:** `Votes` is by far the strongest (shuffling it removes ~0.68 of R²), then `City`, `Longitude`, and the primary cuisine. Without votes, **location dominates** (City ≈ 0.46, Longitude ≈ 0.18, Latitude ≈ 0.10), followed by cuisine, price range and relative cost. Table booking, delivery flags and number of cuisines matter very little.
# - **Interpret with care:** votes are partly a *consequence* of popularity, so they signal quality rather than cause it; and "City" effects partly reflect how restaurants were sampled (see Task 4), not food quality.

print("\nDone. All outputs were saved in:", OUT)
