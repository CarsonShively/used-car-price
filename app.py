import types, sys
import pandas as pd
import joblib
import re
import gradio as gr
import numpy as np
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator
from huggingface_hub import hf_hub_download
from dataclasses import dataclass
import xgboost as xgb
import os

state = types.SimpleNamespace()

REPO_ID = "Carson-Shively/used-car-price"
REV = "main" 
MODEL_PKL_PATH  = "artifacts/used_cars_pipeline.joblib"

def to_str_no_nan(X):
    if hasattr(X, "fillna"):
        return X.fillna("").astype(str)
    return np.asarray(X, dtype=str)

def to_float32(X):
    return X.astype(np.float32)

m = sys.modules.get("__main__") or types.ModuleType("__main__")
sys.modules["__main__"] = m
m.to_str_no_nan = to_str_no_nan
m.to_float32 = to_float32

def load_model_and_schema():
    try:
        pkl_path = hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=MODEL_PKL_PATH,
            revision=REV,
        )
        return joblib.load(pkl_path)
    except Exception as e:
        print(f"Failed to load model from {MODEL_PKL_PATH}")
        print(f"   → repo: {REPO_ID}")
        print(f"   → revision: {REV}")
        print(f"   → error: {e}")
        raise

def init_app():
    bundle = load_model_and_schema()
    state.MODEL = bundle["pipeline"]
    state.META = bundle.get("meta", {})
    state.FEATURE_COLUMNS = tuple(state.META.get("feature_names", []))

    state.PRE = state.MODEL.named_steps["pre"]
    est = getattr(state.MODEL.named_steps["xgb"], "regressor_", state.MODEL.named_steps["xgb"])
    state.BOOSTER = est.get_booster()
    state.FEAT_NAMES = state.META.get("feature_names") or state.PRE.get_feature_names_out().tolist()

def collect_raw_inputs(model_year, milage, model, engine, transmission,
                       brand, int_color, ext_color, clean_title, accident, fuel_type):
    raw = {
        "model_year": model_year,
        "milage": milage,
        "model": model,
        "engine": engine,
        "transmission": transmission,
        "brand": brand,
        "int_col": int_color,
        "ext_col": ext_color,
        "clean_title": clean_title,
        "accident": accident,
        "fuel_type": fuel_type,
    }
    return raw, "Collected raw inputs. (Not validated yet.)"

def normalize_str(x) -> str:
    s = "" if x is None else str(x)
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

class OnlineRequired(BaseModel):
    model_config = ConfigDict(strict=True)
    model_year: float
    milage: float
    model: str
    engine: str
    transmission: str
    brand: str
    int_col: str
    ext_col: str
    clean_title: str
    accident: str
    fuel_type: str

    @field_validator("brand","model","engine","transmission","ext_col","int_col", mode="before")
    @classmethod
    def _norm_textboxes(cls, v):
        return normalize_str(v)

def make_one_row_df(payload: dict) -> pd.DataFrame:
    return pd.DataFrame([payload])

SENTINELS = {"no answer", "no", "different"}

def to_none(x):
    return None if x in SENTINELS else x

@dataclass
class LocalExplain:
    price_pred: float
    full_df: pd.DataFrame            

def local_explain(row: pd.DataFrame) -> LocalExplain:
    pre     = getattr(state, "PRE", state.MODEL.named_steps["pre"])
    booster = getattr(state, "BOOSTER", None)
    if booster is None:
        xgb_step = state.MODEL.named_steps["xgb"]
        est = getattr(xgb_step, "regressor_", xgb_step)
        booster = est.get_booster()
        state.BOOSTER = booster
    names   = getattr(state, "FEAT_NAMES", pre.get_feature_names_out().tolist())

    X_tr = pre.transform(row)

    contribs = booster.predict(
        xgb.DMatrix(X_tr, feature_names=names),
        pred_contribs=True,
        validate_features=False  
    )[0]

    bias, shap_vals = float(contribs[-1]), contribs[:-1]

    assert len(names) == shap_vals.shape[0], "feature_names != SHAP length"

    price_pred = float(np.expm1(bias + float(np.sum(shap_vals))))

    names_arr = np.asarray(names, dtype=object)
    shap_arr  = shap_vals.astype(float, copy=False)

    pct = (np.expm1(shap_arr) * 100.0).astype(float, copy=False)

    full_df = pd.DataFrame({
        "feature":    names_arr.astype(str),
        "shap_log":   shap_arr,
        "pct_effect": pct,
    })
    return LocalExplain(price_pred=price_pred, full_df=full_df)

_TFIDF_PREFIXES = (
    "tfidf_model__",
    "tfidf_engine__",
    "tfidf_transmission__",
)

def aggregate_text_fields(full_df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for prefix, label in [
        ("tfidf_model__", "model_text"),
        ("tfidf_engine__", "engine_text"),
        ("tfidf_transmission__", "transmission_text"),
    ]:
        g = full_df[full_df["feature"].str.startswith(prefix)]
        if not g.empty:
            shap_log_sum = float(g["shap_log"].sum())
            pct_effect   = float(np.expm1(shap_log_sum) * 100.0)
            out.append({"feature": label, "shap_log": shap_log_sum, "pct_effect": pct_effect})
    return pd.DataFrame(out, columns=["feature", "shap_log", "pct_effect"])

def aggregate_ohe(full_df: pd.DataFrame,
                  bases=("fuel_type", "accident", "clean_title")) -> pd.DataFrame:
    keep = full_df.copy()
    agg_rows = []
    for base in bases:
        mask = keep["feature"].str.startswith(f"ohe__{base}_")
        if mask.any():
            shap_log_sum = float(keep.loc[mask, "shap_log"].sum())
            pct_effect   = float(np.expm1(shap_log_sum) * 100.0)
            agg_rows.append({"feature": base, "shap_log": shap_log_sum, "pct_effect": pct_effect})
            keep = keep.loc[~mask].copy()
    if agg_rows:
        return pd.concat([keep, pd.DataFrame(agg_rows)], ignore_index=True)
    return keep

def full_df_with_text_aggregated(full_df: pd.DataFrame) -> pd.DataFrame:
    non_text = ~full_df["feature"].str.startswith(_TFIDF_PREFIXES)
    return pd.concat([full_df[non_text].copy(), aggregate_text_fields(full_df)], ignore_index=True)

FEATURE_LABELS = {
    "num__model_year": "Model Year",
    "num__milage":     "Milage",
    "te__brand":   "Car Brand",
    "te__int_col": "Interior Color",
    "te__ext_col": "Exterior Color",
    "fuel_type":    "Fuel Type",
    "clean_title":  "Clean Title",
    "accident":     "Accident",
    "model_text":        "Model",
    "engine_text":       "Engine",
    "transmission_text": "Transmission",
}

def top_k_features_labeled(full_df: pd.DataFrame, k: int = 5) -> pd.DataFrame:
    ranked = full_df.sort_values("pct_effect", key=lambda s: s.abs(), ascending=False).copy()
    ranked["Feature"] = ranked["feature"].map(FEATURE_LABELS)
    ranked = ranked[ranked["Feature"].notna()].copy()
    ranked["% Effect"] = ranked["pct_effect"].map(lambda v: f"{v:+.1f}%")
    return ranked.loc[:, ["Feature", "% Effect"]].head(k).reset_index(drop=True)

def predict_from_raw(model_year, milage, model, engine, transmission,
                     brand, int_color, ext_color, clean_title, accident, fuel_type) -> tuple[str, pd.DataFrame]:
    raw, _ = collect_raw_inputs(
        model_year, milage, model, engine, transmission,
        brand, int_color, ext_color, clean_title, accident, fuel_type
    )
    try:
        payload = OnlineRequired.model_validate(raw)
    except ValidationError as e:
        raise gr.Error(e.errors()[0]["msg"])

    row = make_one_row_df(payload.model_dump())

    for c in ["fuel_type", "clean_title", "accident"]:
        row[c] = row[c].map(to_none)

    ex = local_explain(row)

    full_agg = full_df_with_text_aggregated(ex.full_df)
    full_agg = aggregate_ohe(full_agg, bases=("fuel_type", "accident", "clean_title"))
    top5     = top_k_features_labeled(full_agg, k=5)

    return f"${ex.price_pred:,.2f}", top5

def predict(model_year, milage, model, engine, transmission,
            brand, int_color, ext_color, clean_title, accident, fuel_type):
    return predict_from_raw(
        model_year, milage, model, engine, transmission,
        brand, int_color, ext_color, clean_title, accident, fuel_type
    )

with gr.Blocks() as demo:
    gr.Markdown("## Used Car Price")

    with gr.Row():
        with gr.Column():
            brand = gr.Textbox(label="Car Brand", placeholder="Lexus")
            model = gr.Textbox(label="Car Model", placeholder="RX 350 RX 350")
            model_year = gr.Number(label="Model Year", minimum=1990, maximum=2025, step=1, value=2018)
            milage = gr.Number(label="Milage", minimum=0, maximum=400000, step=1)
            engine = gr.Textbox(label="Engine", placeholder="3.5 Liter DOHC")
            transmission = gr.Textbox(label="Transmission", placeholder="Automatic")
            fuel_type = gr.Dropdown(["gasoline","hybrid","diesel","e85 flex fuel","plug-in hybrid","different"], label="Fuel Type")
            ext_color = gr.Textbox(label="Exterior Color", placeholder="Black")
            int_color = gr.Textbox(label="Interior Color", placeholder="Blue")
            clean_title = gr.Radio(["yes","no"], label="Clean Title")
            accident = gr.Dropdown(["none reported","at least 1 accident or damage reported","no answer"], label="Accident")
    
    submit = gr.Button("Predict Car Price")
    out_price = gr.Textbox(label="Predicted price")
    out_top5  = gr.Dataframe(label="Top 5 features (by % effect)")
    
    submit.click(
        fn=predict, 
        inputs=[model_year, milage, model, engine, transmission, brand, int_color, ext_color, clean_title, accident, fuel_type],
        outputs=[out_price, out_top5]
    )

if __name__ == "__main__":
    init_app()

    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        show_error=True,
    )