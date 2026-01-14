import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from xgboost import XGBRegressor
from category_encoders import TargetEncoder
from huggingface_hub import hf_hub_download

train_data = hf_hub_download(
        repo_id="Carson-Shively/used-car-price",
        filename="data/silver/silver.parquet",
        repo_type="dataset",
        revision="main",
    )

df = pd.read_parquet(train_data)

y_log = np.log1p(df["price"]).astype(np.float32)
X = df.drop(columns=["price"]).copy()

for col in ["model", "engine", "transmission"]:
    X[col] = X[col].fillna("").astype(str)

X["accident_flag"] = X["accident"].map({
    "at least 1 accident or damage reported": 1,
    "none reported": 0
}).astype("float32")

X["clean_title_flag"] = X["clean_title"].map({"yes": 1}).astype("float32")

NUMERIC = ["model_year", "milage", "accident_flag", "clean_title_flag"]
TARGET_ENCODE = ["brand", "int_col", "ext_col"]
cv = KFold(n_splits=5, shuffle=True, random_state=42)

def make_preprocessor(trial):
    m_an = trial.suggest_categorical("model.analyzer", ["word", "char_wb"])
    m_ng_word = trial.suggest_categorical("model.ngram_word", [(1,1), (1,2), (1,3)])
    m_ng_char = trial.suggest_categorical("model.ngram_char", [(3,5), (4,6), (5,7)])
    m_ng = m_ng_word if m_an == "word" else m_ng_char
    m_tok = r"[A-Za-z0-9][A-Za-z0-9\-']+" if m_an == "word" else None

    tfidf_model = TfidfVectorizer(
        analyzer=m_an, ngram_range=m_ng, token_pattern=m_tok,
        min_df=trial.suggest_float("model.min_df", 0.001, 0.02, log=True),
        max_df=trial.suggest_float("model.max_df", 0.85, 1.0),
        max_features=trial.suggest_categorical("model.max_features", [10000, 30000, 60000, None]),
        sublinear_tf=trial.suggest_categorical("model.sublinear_tf", [True, False]),
        norm=trial.suggest_categorical("model.norm", ["l2", None]),
        strip_accents="unicode", lowercase=True, dtype=np.float32
    )

    e_an = trial.suggest_categorical("engine.analyzer", ["word", "char_wb"])
    e_ng_word = trial.suggest_categorical("engine.ngram_word", [(1,1), (1,2), (1,3)])
    e_ng_char = trial.suggest_categorical("engine.ngram_char", [(3,5), (4,6), (5,7)])
    e_ng = e_ng_word if e_an == "word" else e_ng_char
    e_tok = r"[A-Za-z0-9][A-Za-z0-9\.\-']+" if e_an == "word" else None

    tfidf_engine = TfidfVectorizer(
        analyzer=e_an, ngram_range=e_ng, token_pattern=e_tok,
        min_df=trial.suggest_float("engine.min_df", 0.001, 0.02, log=True),
        max_df=trial.suggest_float("engine.max_df", 0.85, 1.0),
        max_features=trial.suggest_categorical("engine.max_features", [5000, 20000, 40000, None]),
        sublinear_tf=trial.suggest_categorical("engine.sublinear_tf", [True, False]),
        norm=trial.suggest_categorical("engine.norm", ["l2", None]),
        strip_accents="unicode", lowercase=True, dtype=np.float32
    )

    t_an = trial.suggest_categorical("trans.analyzer", ["word", "char_wb"])
    t_ng_word = trial.suggest_categorical("trans.ngram_word", [(1,1), (1,2)])
    t_ng_char = trial.suggest_categorical("trans.ngram_char", [(3,5), (4,6)])
    t_ng = t_ng_word if t_an == "word" else t_ng_char
    t_tok = r"[A-Za-z0-9][A-Za-z0-9\-']+" if t_an == "word" else None

    tfidf_trans = TfidfVectorizer(
        analyzer=t_an, ngram_range=t_ng, token_pattern=t_tok,
        min_df=trial.suggest_float("trans.min_df", 0.001, 0.02, log=True),
        max_df=trial.suggest_float("trans.max_df", 0.85, 1.0),
        max_features=trial.suggest_categorical("trans.max_features", [2000, 10000, 20000, None]),
        sublinear_tf=trial.suggest_categorical("trans.sublinear_tf", [True, False]),
        norm=trial.suggest_categorical("trans.norm", ["l2", None]),
        strip_accents="unicode", lowercase=True, dtype=np.float32
    )

    te_smoothing = trial.suggest_float("te.smoothing", 2.0, 50.0, log=True)
    te_minleaf  = trial.suggest_int("te.min_samples_leaf", 1, 100)

    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf_model", tfidf_model, "model"),
            ("tfidf_engine", tfidf_engine, "engine"),
            ("tfidf_transmission", tfidf_trans, "transmission"),
            ("num", FunctionTransformer(lambda X: X.astype(np.float32), accept_sparse=True), NUMERIC),
            ("te", TargetEncoder(cols=None, smoothing=te_smoothing, min_samples_leaf=te_minleaf,
                                 handle_unknown="value", handle_missing="value"), TARGET_ENCODE),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True), ["fuel_type"]),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
        sparse_threshold=1.0
    )
    return preprocessor

def make_model(trial):
    return XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_estimators=trial.suggest_int("xgb.n_estimators", 400, 1800, step=200),
        learning_rate=trial.suggest_float("xgb.learning_rate", 0.01, 0.2, log=True),
        max_depth=trial.suggest_int("xgb.max_depth", 4, 10),
        min_child_weight=trial.suggest_float("xgb.min_child_weight", 1e-3, 20.0, log=True),
        subsample=trial.suggest_float("xgb.subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("xgb.colsample_bytree", 0.6, 1.0),
        reg_alpha=trial.suggest_float("xgb.reg_alpha", 1e-6, 1.0, log=True),
        reg_lambda=trial.suggest_float("xgb.reg_lambda", 1e-3, 10.0, log=True),
        gamma=trial.suggest_float("xgb.gamma", 0.0, 1.5),
        n_jobs=-1
    )

def objective(trial):
    pre = make_preprocessor(trial)
    xgb = make_model(trial)
    pipe = Pipeline([("pre", pre), ("xgb", xgb)])
    scores = cross_val_score(pipe, X, y_log, cv=cv,
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
    return -scores.mean()

sampler = optuna.samplers.TPESampler(seed=42)
pruner = optuna.pruners.MedianPruner(n_warmup_steps=10)
study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

N_TRIALS = 150
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

best = study.best_trial.params

def _grp(prefix):
    return {k.split(".",1)[1]: v for k, v in best.items() if k.startswith(prefix)}

print("\nBest params by block:")
print("TFIDF model:", _grp("model"))
print("TFIDF engine:", _grp("engine"))
print("TFIDF transmission:", _grp("trans"))
print("XGB:", _grp("xgb"))

print("\nBest CV RMSE on y_log (RMSLE):", study.best_value)