import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, TargetEncoder
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_squared_log_error, mean_squared_error, mean_absolute_error, r2_score

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
cv = KFold(n_splits=5, shuffle=True, random_state=42)

y_pred_oof = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1, method="predict")

y_true = y.astype(float).to_numpy()
y_pred = y_pred_oof.astype(float)

rmsle = mean_squared_log_error(y_true, y_pred) ** 0.5
rmse  = mean_squared_error(y_true, y_pred) ** 0.5
mae   = mean_absolute_error(y_true, y_pred)
r2    = r2_score(y_true, y_pred)

print(f"RMSLE: {rmsle:.4f}")
print(f"RMSE:  {rmse:,.2f}")
print(f"MAE:   {mae:,.2f}")
print(f"R^2:   {r2:.4f}")