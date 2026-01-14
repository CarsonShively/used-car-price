import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, TargetEncoder
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import joblib
from pathlib import Path
from huggingface_hub import hf_hub_download

train_data = hf_hub_download(
        repo_id="Carson-Shively/used-car-price",
        filename="data/silver/silver.parquet",
        repo_type="dataset",
        revision="main",
    )

df = pd.read_parquet(train_data)

y = df["price"].astype(float)
X = df.drop(columns=["price"])

NUMERIC = ["model_year", "milage"]

TARGET_ENCODE = ["brand", "int_col", "ext_col"]

OHE = ["fuel_type", "accident", "clean_title"]

def to_str_no_nan(X):
    if hasattr(X, "fillna"):
        return X.fillna("").astype(str)
    return np.asarray(X, dtype=str)

def to_float32(X):
    if hasattr(X, "apply"):
        return X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    return pd.to_numeric(X, errors="coerce").astype(np.float32)

tfidf_model = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=0.0019354710047961533,
    max_df=0.929823623281149,
    max_features=None,
    sublinear_tf=False,
    norm=None,
    strip_accents="unicode",
    lowercase=True,
    dtype=np.float32
)
model_pipe = Pipeline([
    ("to_str", FunctionTransformer(to_str_no_nan, feature_names_out="one-to-one")),
    ("tfidf",  tfidf_model),
])

tfidf_engine = TfidfVectorizer(
    analyzer="char_wb",
    ngram_range=(3, 5),
    min_df=0.009150665876378068,
    max_df=0.9793563048246156,
    max_features=40000,
    sublinear_tf=True,
    norm=None,
    strip_accents="unicode",
    lowercase=True,
    dtype=np.float32
)
engine_pipe = Pipeline([
    ("to_str", FunctionTransformer(to_str_no_nan, feature_names_out="one-to-one")),
    ("tfidf",  tfidf_engine),
])

tfidf_transmission = TfidfVectorizer(
    analyzer="word",
    ngram_range=(1, 2),
    token_pattern=r"[A-Za-z0-9][A-Za-z0-9\-']+",
    min_df=0.019434358594147716,
    max_df=0.9130841255114773,
    max_features=2000,
    sublinear_tf=False,
    norm="l2",
    strip_accents="unicode",
    lowercase=True,
    dtype=np.float32
)
transmission_pipe = Pipeline([
    ("to_str", FunctionTransformer(to_str_no_nan, feature_names_out="one-to-one")),
    ("tfidf",  tfidf_transmission),
])

te = TargetEncoder(target_type="continuous")

ohe = OneHotEncoder(drop="if_binary", handle_unknown="ignore", sparse_output=True)

num_ft = FunctionTransformer(
    to_float32,
    accept_sparse=True,
    feature_names_out="one-to-one"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("tfidf_model", model_pipe, "model"),
        ("tfidf_engine", engine_pipe, "engine"),
        ("tfidf_transmission", transmission_pipe, "transmission"),
        ("num", num_ft, NUMERIC),
        ("te", te, TARGET_ENCODE),
        ("ohe", ohe, OHE),
    ],
    remainder="drop",
    verbose_feature_names_out=True,
    sparse_threshold=1.0
)

xgb = XGBRegressor(
    objective="reg:squarederror",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
    n_estimators=1600,
    learning_rate=0.04623222045025156,
    max_depth=5,
    min_child_weight=0.013683558256795545,
    subsample=0.7909332905323827,
    colsample_bytree=0.7843244316811016,
    reg_alpha=4.739492673356427e-05,
    reg_lambda=0.02532290345612286,
    gamma=0.002125732013130148
)

model = TransformedTargetRegressor(
    regressor=xgb,
    func=np.log1p,
    inverse_func=np.expm1
)

pipeline = Pipeline([
    ("pre", preprocessor),
    ("xgb", model),
])

pipeline.fit(X, y)
y_pred = pipeline.predict(X)

pre = pipeline.named_steps["pre"]
ttr = pipeline.named_steps["xgb"]
xgb_est = ttr.regressor

feat_names = pre.get_feature_names_out().tolist()

def group_of(name: str) -> str:
    if name.startswith("tfidf_model__"):        return "model_text"
    if name.startswith("tfidf_engine__"):       return "engine_text"
    if name.startswith("tfidf_transmission__"): return "transmission_text"
    if name.startswith("num__"):                return name.split("__", 1)[-1]
    if name.startswith("te__"):                 return name.split("__", 1)[-1] + "_TE"
    if name.startswith("ohe__"):                return "fuel_type"
    return "other"

group_map = {i: group_of(n) for i, n in enumerate(feat_names)}
token_map = {i: n.split("__", 1)[-1] for i, n in enumerate(feat_names) if n.startswith("tfidf_")}

meta = {
    "group_map": group_map,
    "token_map": token_map,
    "log_target": True,
    "feature_names": feat_names,
}

Path("artifacts").mkdir(exist_ok=True)
joblib.dump({"pipeline": pipeline, "meta": meta},
            "artifacts/used_cars_pipeline.joblib",
            compress=("xz", 3))
