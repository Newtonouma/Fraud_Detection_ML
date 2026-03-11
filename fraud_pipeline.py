"""
fraud_pipeline.py  —  v2.0
===========================
Revised pipeline for the AI Fraud Detector .exe

Key changes in v2:
  1. Pre-trains on merged_fraud_dataset.csv at startup (bundled in exe)
  2. Universal column mapper — accepts ANY csv/xlsx/json regardless of
     column names and intelligently maps to the 52-feature space
  3. Inference mode — loaded external files are scored by the pre-trained
     model without retraining
  4. Training mode — user can still retrain on any properly labelled file

University of the West of Scotland — MSc Project
Evans Polley | B01823633
"""

import os, sys, warnings, json
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# ──────────────────────────────────────────────────────────────
#  52 UNIFIED FEATURE COLUMNS  (fixed — must never change order)
# ──────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "amt", "trans_hour", "trans_dow", "trans_month", "trans_day",
    "city_pop", "age", "geo_distance", "is_online", "amt_zscore",
    "cat_food_dining", "cat_gas_transport", "cat_grocery_net", "cat_grocery_pos",
    "cat_health_fitness", "cat_home", "cat_kids_pets", "cat_misc_net",
    "cat_misc_pos", "cat_personal_care", "cat_shopping_net",
    "cat_shopping_pos", "cat_travel", "cat_entertainment",
    "v1","v2","v3","v4","v5","v6","v7","v8","v9","v10",
    "v11","v12","v13","v14","v15","v16","v17","v18","v19","v20",
    "v21","v22","v23","v24","v25","v26","v27","v28",
]

# ──────────────────────────────────────────────────────────────
#  RESOURCE PATH  (handles both .py and PyInstaller .exe)
# ──────────────────────────────────────────────────────────────
def resource_path(relative: str) -> str:
    """Return absolute path — works both in dev and inside .exe bundle."""
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, relative)


# ──────────────────────────────────────────────────────────────
#  UNIVERSAL COLUMN MAPPER
#  Maps ANY incoming DataFrame columns → FEATURE_COLS space
# ──────────────────────────────────────────────────────────────

# Keyword-based fuzzy mapping: if any keyword appears in the column name
# (after lowercasing / stripping) → map to the target feature
COLUMN_KEYWORD_MAP = {
    # Amount
    "amt":            ["amt", "amount", "transaction_amount", "trans_amount",
                       "value", "purchase_amount", "sum", "price"],
    # Temporal
    "trans_hour":     ["hour", "trans_hour", "time_hour"],
    "trans_dow":      ["dow", "day_of_week", "weekday", "dayofweek"],
    "trans_month":    ["month", "trans_month"],
    "trans_day":      ["day", "trans_day", "day_of_month"],
    # Demographics / Geography
    "city_pop":       ["city_pop", "population", "pop", "city_population"],
    "age":            ["age", "customer_age", "cardholder_age", "holder_age"],
    "geo_distance":   ["geo_distance", "distance", "dist", "geo_dist",
                       "merchant_distance"],
    "is_online":      ["is_online", "online", "channel", "is_net",
                       "ecommerce", "internet"],
    "amt_zscore":     ["amt_zscore", "amount_zscore", "zscore", "amount_z"],
    # Categories (exact match preferred, then keyword)
    "cat_food_dining":    ["food_dining", "food", "dining", "restaurant"],
    "cat_gas_transport":  ["gas_transport", "gas", "fuel", "transport"],
    "cat_grocery_net":    ["grocery_net", "grocery_online"],
    "cat_grocery_pos":    ["grocery_pos", "grocery"],
    "cat_health_fitness": ["health_fitness", "health", "fitness", "medical"],
    "cat_home":           ["home", "household"],
    "cat_kids_pets":      ["kids_pets", "kids", "pets", "children"],
    "cat_misc_net":       ["misc_net"],
    "cat_misc_pos":       ["misc_pos", "misc", "miscellaneous"],
    "cat_personal_care":  ["personal_care", "personal", "beauty", "care"],
    "cat_shopping_net":   ["shopping_net", "online_shopping"],
    "cat_shopping_pos":   ["shopping_pos", "shopping", "retail"],
    "cat_travel":         ["travel", "airline", "hotel", "tourism"],
    "cat_entertainment":  ["entertainment", "leisure", "gaming", "cinema"],
    # PCA V-features (European dataset — direct name match only)
    **{f"v{i}": [f"v{i}"] for i in range(1, 29)},
}

# Columns that can be used to DERIVE features if direct mapping not found
DERIVABLE = {
    # If we have a raw datetime string → derive trans_hour, dow, month, day
    "datetime_raw": ["trans_date_trans_time", "datetime", "timestamp",
                     "date", "transaction_date", "trans_date", "time"],
    # If we have lat/long and merch_lat/long → derive geo_distance
    "lat":          ["lat", "latitude", "cardholder_lat", "customer_lat"],
    "long":         ["long", "longitude", "cardholder_long", "customer_long"],
    "merch_lat":    ["merch_lat", "merchant_lat", "merchant_latitude"],
    "merch_long":   ["merch_long", "merchant_long", "merchant_longitude"],
    # If we have dob → derive age
    "dob":          ["dob", "date_of_birth", "birth_date", "birthdate"],
    # Raw category string → derive cat_* OHE flags
    "category_raw": ["category", "merchant_category", "mcc", "cat",
                     "transaction_category", "type"],
    # Fraud label
    "is_fraud":     ["is_fraud", "fraud", "class", "label", "target",
                     "fraudulent", "is_fraudulent"],
}

CATEGORY_MAP = {
    "food_dining": "cat_food_dining", "gas_transport": "cat_gas_transport",
    "grocery_net": "cat_grocery_net", "grocery_pos": "cat_grocery_pos",
    "health_fitness": "cat_health_fitness", "home": "cat_home",
    "kids_pets": "cat_kids_pets", "misc_net": "cat_misc_net",
    "misc_pos": "cat_misc_pos", "personal_care": "cat_personal_care",
    "shopping_net": "cat_shopping_net", "shopping_pos": "cat_shopping_pos",
    "travel": "cat_travel", "entertainment": "cat_entertainment",
}


def _normalize_col(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def map_columns(df: pd.DataFrame) -> tuple:
    """
    Map an arbitrary DataFrame's columns to FEATURE_COLS + metadata.

    Returns:
        (mapped_df, mapping_report)
        mapped_df       — DataFrame with exactly FEATURE_COLS + is_fraud + trans_datetime
        mapping_report  — list of strings describing what was mapped/derived/defaulted
    """
    df = df.copy()
    df.columns = [_normalize_col(c) for c in df.columns]
    report = []
    out = pd.DataFrame(index=df.index)

    col_set = set(df.columns)

    # ── Step 1: Resolve special derivable columns ──
    datetime_col = _find_col(df, DERIVABLE["datetime_raw"])
    dob_col      = _find_col(df, DERIVABLE["dob"])
    cat_col      = _find_col(df, DERIVABLE["category_raw"])
    lat_col      = _find_col(df, DERIVABLE["lat"])
    lon_col      = _find_col(df, DERIVABLE["long"])
    mlat_col     = _find_col(df, DERIVABLE["merch_lat"])
    mlon_col     = _find_col(df, DERIVABLE["merch_long"])
    fraud_col    = _find_col(df, DERIVABLE["is_fraud"])

    # ── Step 2: Datetime derivation ──
    if datetime_col:
        dt = pd.to_datetime(df[datetime_col], utc=True, errors="coerce")
        df["_dt"] = dt
        report.append(f"✅ Datetime derived from '{datetime_col}'")
    else:
        df["_dt"] = pd.NaT

    # ── Step 3: Age derivation ──
    if dob_col:
        try:
            dob = pd.to_datetime(df[dob_col], utc=True, errors="coerce")
            df["_age"] = ((pd.Timestamp("2020-01-01", tz="UTC") - dob)
                          .dt.days / 365.25).clip(18, 100)
            report.append(f"✅ Age derived from '{dob_col}'")
        except Exception:
            df["_age"] = 40.0
    else:
        df["_age"] = 40.0

    # ── Step 4: Geo distance derivation ──
    if lat_col and lon_col and mlat_col and mlon_col:
        df["_geo"] = np.sqrt(
            (df[lat_col].astype(float) - df[mlat_col].astype(float))**2 +
            (df[lon_col].astype(float) - df[mlon_col].astype(float))**2
        ).fillna(0)
        report.append(f"✅ Geo distance derived from lat/long columns")
    else:
        df["_geo"] = 0.0

    # ── Step 5: Category OHE derivation ──
    cat_flags_derived = False
    if cat_col:
        cat_vals = df[cat_col].astype(str).str.lower().str.strip()
        for raw, col_name in CATEGORY_MAP.items():
            df[col_name] = (cat_vals == raw).astype(int)
        # is_online from category
        df["_is_online_cat"] = cat_vals.str.endswith("_net").astype(int)
        cat_flags_derived = True
        report.append(f"✅ Category flags derived from '{cat_col}'")

    # ── Step 6: Amount zscore ──
    amt_src = _find_col(df, COLUMN_KEYWORD_MAP["amt"])
    if amt_src:
        df["_amt"] = pd.to_numeric(df[amt_src], errors="coerce").fillna(0)
        std = df["_amt"].std() + 1e-9
        df["_amt_zscore"] = ((df["_amt"] - df["_amt"].mean()) / std)
    else:
        df["_amt"] = 0.0
        df["_amt_zscore"] = 0.0

    # ── Step 7: Map every FEATURE_COL ──
    mapped = 0
    derived = 0
    defaulted = 0

    for feat in FEATURE_COLS:
        # Direct keyword match
        src = _find_col(df, COLUMN_KEYWORD_MAP.get(feat, [feat]))
        if src and src in col_set:
            out[feat] = pd.to_numeric(df[src], errors="coerce").fillna(0)
            mapped += 1
            continue

        # Derived substitutes
        if feat == "amt":
            out[feat] = df["_amt"]; derived += 1; continue
        if feat == "amt_zscore":
            out[feat] = df["_amt_zscore"]; derived += 1; continue
        if feat == "age":
            out[feat] = df["_age"]; derived += 1; continue
        if feat == "geo_distance":
            out[feat] = df["_geo"]; derived += 1; continue
        if feat == "trans_hour" and "_dt" in df.columns and df["_dt"].notna().any():
            out[feat] = df["_dt"].dt.hour.fillna(0).astype(float); derived += 1; continue
        if feat == "trans_dow" and "_dt" in df.columns and df["_dt"].notna().any():
            out[feat] = df["_dt"].dt.dayofweek.fillna(0).astype(float); derived += 1; continue
        if feat == "trans_month" and "_dt" in df.columns and df["_dt"].notna().any():
            out[feat] = df["_dt"].dt.month.fillna(1).astype(float); derived += 1; continue
        if feat == "trans_day" and "_dt" in df.columns and df["_dt"].notna().any():
            out[feat] = df["_dt"].dt.day.fillna(1).astype(float); derived += 1; continue
        if feat.startswith("cat_") and cat_flags_derived and feat in df.columns:
            out[feat] = df[feat].astype(float); derived += 1; continue
        if feat == "is_online":
            if cat_flags_derived and "_is_online_cat" in df.columns:
                out[feat] = df["_is_online_cat"].astype(float); derived += 1; continue
            else:
                out[feat] = 0.0; defaulted += 1; continue

        # Default to 0
        out[feat] = 0.0
        defaulted += 1

    report.append(
        f"📊 Columns: {mapped} directly mapped  |  "
        f"{derived} derived  |  {defaulted} defaulted to 0"
    )

    # ── Step 8: Target column ──
    if fraud_col:
        out["is_fraud"] = pd.to_numeric(df[fraud_col], errors="coerce").fillna(-1)
        report.append(f"✅ Label column found: '{fraud_col}'")
    else:
        out["is_fraud"] = -1   # -1 = unknown (inference mode)
        report.append("ℹ️  No label column found — running in inference mode")

    # ── Step 9: Metadata ──
    out["trans_datetime"] = df["_dt"] if "_dt" in df.columns else pd.NaT

    # Preserve original columns for display
    out["_raw_df"] = None   # placeholder — actual raw kept separately

    return out.drop(columns=["_raw_df"]), report


def _find_col(df: pd.DataFrame, keywords: list):
    """Return the first df column whose normalised name matches any keyword."""
    col_lower = {c: c for c in df.columns}  # already normalised
    for kw in keywords:
        kw_norm = _normalize_col(kw)
        if kw_norm in col_lower:
            return col_lower[kw_norm]
        # Partial match
        for col in df.columns:
            if kw_norm in col or col in kw_norm:
                return col
    return None


# ──────────────────────────────────────────────────────────────
#  FRAUD DATA PIPELINE
# ──────────────────────────────────────────────────────────────

class FraudDataPipeline:
    """
    v2 Pipeline — two modes:

    TRAINING MODE (on merged_fraud_dataset.csv):
        p = FraudDataPipeline()
        p.load_training_data()      ← loads bundled merged CSV
        p.split_and_resample()
        # → pass to ModelManager.train_all()

    INFERENCE MODE (any new file):
        p.load_external(filepath)   ← maps ANY columns → feature space
        # → pass to ModelManager.predict_all()
    """

    BUNDLED_DATASET = "merged_fraud_dataset.csv"

    def __init__(self):
        self.df          = None    # current active DataFrame (unified cols)
        self.raw_df      = None    # original columns for display in table
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.scaler      = StandardScaler()
        self.imputer     = SimpleImputer(strategy="median")
        self.load_report = []
        self.mode        = None    # "training" | "inference"
        self.has_labels  = False

    # ── Load bundled training data ──
    def load_training_data(self) -> pd.DataFrame:
        path = resource_path(self.BUNDLED_DATASET)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Bundled training dataset not found at: {path}\n"
                "Ensure merged_fraud_dataset.csv is in the same folder as app.py"
            )
        df = pd.read_csv(path, low_memory=False)
        df = df.dropna(subset=["is_fraud"])
        # Already in unified feature space
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        self.df         = df
        self.raw_df     = df.copy()
        self.mode       = "training"
        self.has_labels = True
        self.load_report = [
            f"✅ Bundled training dataset loaded",
            f"   Rows: {len(df):,}  |  "
            f"Fraud: {int(df['is_fraud'].sum()):,} "
            f"({df['is_fraud'].mean()*100:.3f}%)",
            f"   Sources: {', '.join(df['source_tag'].unique()) if 'source_tag' in df.columns else 'N/A'}",
        ]
        return df

    # ── Load any external file (inference) ──
    def load_external(self, filepath: str) -> pd.DataFrame:
        raw = self._read_file(filepath)
        self.raw_df = raw.copy()

        mapped, mapping_report = map_columns(raw)
        self.df         = mapped
        self.mode       = "inference"
        self.has_labels = (mapped["is_fraud"] >= 0).any()
        self.load_report = [
            f"✅ Loaded: {os.path.basename(filepath)}",
            f"   Original columns: {len(raw.columns)}  |  Rows: {len(raw):,}",
        ] + mapping_report
        if self.has_labels:
            valid_labels = mapped[mapped["is_fraud"] >= 0]["is_fraud"]
            self.load_report.append(
                f"   Labelled rows: {len(valid_labels):,}  |  "
                f"Fraud: {int((valid_labels==1).sum()):,} "
                f"({valid_labels.mean()*100:.3f}%)"
            )
        return mapped

    # ── Backwards-compatible single loader ──
    def load(self, filepath: str) -> pd.DataFrame:
        return self.load_external(filepath)

    # ── Train/test split + SMOTE (training mode only) ──
    def split_and_resample(self, test_size: float = 0.2,
                            use_smote: bool = True) -> tuple:
        if self.df is None:
            raise ValueError("Load data first.")
        df = self.df[self.df["is_fraud"] >= 0].copy()
        X  = df[FEATURE_COLS].values.astype(float)
        y  = df["is_fraud"].values.astype(int)

        X = self.imputer.fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42)
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        if use_smote and SMOTE_AVAILABLE and y_tr.sum() >= 6:
            try:
                k = min(5, y_tr.sum() - 1)
                sm = SMOTE(random_state=42, k_neighbors=k)
                X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
                self.load_report.append(
                    f"🔁 SMOTE: train balanced → "
                    f"{int(y_tr.sum()):,} fraud / {int((y_tr==0).sum()):,} legit")
            except Exception as e:
                self.load_report.append(f"⚠️ SMOTE skipped: {e}")

        self.X_train, self.X_test = X_tr, X_te
        self.y_train, self.y_test = y_tr, y_te
        return X_tr, X_te, y_tr, y_te

    # ── Get inference-ready feature matrix ──
    def get_inference_X(self) -> np.ndarray:
        """Return scaled feature matrix for a new (unlabelled) file."""
        if self.df is None:
            raise ValueError("Load a file first.")
        X = self.df[FEATURE_COLS].values.astype(float)
        X = self.imputer.transform(X)   # use fitted imputer
        X = self.scaler.transform(X)    # use fitted scaler
        return X

    def print_report(self):
        print("\n" + "=" * 65)
        print("  FRAUD PIPELINE REPORT")
        print("=" * 65)
        for line in self.load_report:
            print(" ", line)
        print("=" * 65 + "\n")

    def get_source_summary(self) -> pd.DataFrame:
        if self.df is None or "source_tag" not in self.df.columns:
            return pd.DataFrame()
        valid = self.df[self.df["is_fraud"] >= 0]
        return (valid.groupby("source_tag")["is_fraud"]
                .agg(total="count",
                     fraud=lambda x: int((x==1).sum()),
                     legit=lambda x: int((x==0).sum()),
                     fraud_pct=lambda x: f"{x.mean()*100:.3f}%")
                .reset_index())

    @property
    def feature_names(self):
        return FEATURE_COLS

    @staticmethod
    def _read_file(path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[-1].lower()
        if ext == ".csv":
            return pd.read_csv(path, low_memory=False)
        elif ext in (".xlsx", ".xls"):
            return pd.read_excel(path)
        elif ext == ".json":
            return pd.read_json(path)
        raise ValueError(f"Unsupported file type: {ext}")


# ──────────────────────────────────────────────────────────────
#  MODEL MANAGER  — train + predict
# ──────────────────────────────────────────────────────────────

class ModelManager:
    MODELS = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree":       DecisionTreeClassifier(
            max_depth=10, class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            random_state=42, n_jobs=-1),
        "SVM":                 SVC(
            kernel="rbf", class_weight="balanced",
            probability=True, random_state=42),
    }

    def __init__(self):
        self.trained = {}
        self.results = {}

    def is_trained(self) -> bool:
        return len(self.trained) > 0

    def train_all(self, X_train, y_train, X_test, y_test,
                  progress_cb=None) -> dict:
        self.results = {}; self.trained = {}
        for name, model in self.MODELS.items():
            if progress_cb:
                progress_cb(f"Training {name}…")
            try:
                m = model.__class__(**model.get_params())
                m.fit(X_train, y_train)
                y_pred = m.predict(X_test)
                try:
                    auc = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])
                except Exception:
                    auc = 0.0
                self.trained[name] = m
                self.results[name] = {
                    "accuracy":  accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred, zero_division=0),
                    "recall":    recall_score(y_test, y_pred, zero_division=0),
                    "f1":        f1_score(y_test, y_pred, zero_division=0),
                    "auc":       auc,
                    "cm":        confusion_matrix(y_test, y_pred),
                    "y_pred":    y_pred,
                }
            except Exception as e:
                self.results[name] = {"error": str(e)}
        return self.results

    def train_single(self, name, X_train, y_train, X_test, y_test) -> dict:
        m = self.MODELS[name].__class__(**self.MODELS[name].get_params())
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        try:
            auc = roc_auc_score(y_test, m.predict_proba(X_test)[:,1])
        except Exception:
            auc = 0.0
        self.trained[name] = m
        self.results[name] = {
            "accuracy":  accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall":    recall_score(y_test, y_pred, zero_division=0),
            "f1":        f1_score(y_test, y_pred, zero_division=0),
            "auc":       auc,
            "cm":        confusion_matrix(y_test, y_pred),
            "y_pred":    y_pred,
        }
        return self.results[name]

    def predict_all(self, X: np.ndarray,
                    progress_cb=None) -> dict:
        """
        Run inference on a new (potentially unlabelled) dataset.
        Returns dict of {model_name: {"predictions": [...], "probabilities": [...]}}
        """
        if not self.trained:
            raise RuntimeError("No models trained yet. Run detection on the "
                               "training data first.")
        predictions = {}
        for name, m in self.trained.items():
            if progress_cb:
                progress_cb(f"Scoring with {name}…")
            try:
                y_pred = m.predict(X)
                try:
                    y_prob = m.predict_proba(X)[:, 1]
                except Exception:
                    y_prob = y_pred.astype(float)
                predictions[name] = {
                    "predictions":   y_pred,
                    "probabilities": y_prob,
                }
            except Exception as e:
                predictions[name] = {"error": str(e)}
        return predictions

    def predict_best(self, X: np.ndarray) -> np.ndarray:
        """Return predictions from the best-performing trained model (by F1)."""
        if not self.trained:
            raise RuntimeError("Train models first.")
        valid = {k: v for k, v in self.results.items() if "f1" in v}
        if valid:
            best = max(valid, key=lambda k: valid[k]["f1"])
        else:
            best = list(self.trained.keys())[0]
        return self.trained[best].predict(X)
