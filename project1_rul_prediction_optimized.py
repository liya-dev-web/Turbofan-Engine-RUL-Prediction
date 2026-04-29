
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import joblib
import os

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
np.random.seed(42)

# 1. DATA SIMULATION

SENSOR_NAMES = [f"sensor_{i:02d}" for i in range(1, 22)]
SETTING_NAMES = ["op_setting_1", "op_setting_2", "op_setting_3"]
INFORMATIVE_SENSORS = [2, 3, 4, 7, 8, 9, 11, 12, 13, 14, 15, 17, 20, 21]

def simulate_engine_data(n_engines: int = 100,
                          min_life: int = 128,
                          max_life: int = 362) -> pd.DataFrame:
    records = []
    for unit in range(1, n_engines + 1):
        max_cycle = np.random.randint(min_life, max_life)
        for cycle in range(1, max_cycle + 1):
            degradation = cycle / max_cycle
            row = {"unit_id": unit, "cycle": cycle}
            row["op_setting_1"] = round(np.random.choice([0.0, 0.25, 0.42, 0.84, 1.0]) +
                                        np.random.normal(0, 0.001), 4)
            row["op_setting_2"] = round(np.random.choice([0.0, 0.0003, 0.0004]) +
                                        np.random.normal(0, 0.00001), 6)
            row["op_setting_3"] = int(np.random.choice([60, 80, 100]))
            for s in range(1, 22):
                if s in INFORMATIVE_SENSORS:
                    drift = degradation * np.random.uniform(0.05, 0.20)
                    direction = 1 if s % 2 == 0 else -1
                    base = np.random.uniform(300, 800) if s > 5 else np.random.uniform(1, 50)
                    value = base + direction * drift * base + np.random.normal(0, base * 0.005)
                else:
                    base = np.random.uniform(1, 50)
                    value = base + np.random.normal(0, base * 0.002)
                row[f"sensor_{s:02d}"] = round(value, 4)
            records.append(row)
    return pd.DataFrame(records)

def compute_rul(df: pd.DataFrame, clip_max: int = 130) -> pd.DataFrame:
    max_cycles = df.groupby("unit_id")["cycle"].max().rename("max_cycle")
    df = df.join(max_cycles, on="unit_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul"] = df["rul"].clip(upper=clip_max)
    df.drop(columns=["max_cycle"], inplace=True)
    return df

# 2. FEATURE ENGINEERING


def engineer_features(df: pd.DataFrame, window: int = 15) -> pd.DataFrame:
    df = df.sort_values(["unit_id", "cycle"]).copy()
    new_cols = {}
    for col in SENSOR_NAMES:
        grp = df.groupby("unit_id")[col]
        new_cols[f"{col}_roll_mean"] = grp.transform(lambda x: x.rolling(window, min_periods=1).mean())
        new_cols[f"{col}_roll_std"] = grp.transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
        new_cols[f"{col}_ewm"] = grp.transform(lambda x: x.ewm(span=window, adjust=False).mean())
        new_cols[f"{col}_diff"] = grp.transform(lambda x: x.diff().fillna(0))
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df

def drop_low_variance_sensors(df: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
    sensor_std = df[SENSOR_NAMES].std()
    flat_sensors = sensor_std[sensor_std < threshold].index.tolist()
    df.drop(columns=flat_sensors, inplace=True)
    return df

def unit_train_test_split(df: pd.DataFrame, test_frac: float = 0.2):
    units = df["unit_id"].unique()
    n_test = max(1, int(len(units) * test_frac))
    test_units = np.random.choice(units, size=n_test, replace=False)
    train = df[~df["unit_id"].isin(test_units)].copy()
    test  = df[ df["unit_id"].isin(test_units)].copy()
    return train, test

# 3. EVALUATION

def nasa_score(y_true, y_pred):
    diff = y_pred - y_true
    score = np.where(diff < 0, np.exp(-diff / 13) - 1, np.exp( diff / 10) - 1)
    return np.sum(score)

def evaluate(name, y_true, y_pred):
    rmse  = np.sqrt(mean_squared_error(y_true, y_pred))
    mae   = mean_absolute_error(y_true, y_pred)
    r2    = r2_score(y_true, y_pred)
    score = nasa_score(y_true, y_pred)
    print(f"  {name:<30s}  RMSE={rmse:6.2f}  MAE={mae:6.2f}  R2={r2:.3f}  NASA_Score={score:,.0f}")
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "NASA_Score": score}

# 4. MODELS (OPTIMIZED FOR MULTI-CORE)

def build_models():
    return {
        "Ridge Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  Ridge(alpha=10))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  RandomForestRegressor(
                n_estimators=100, max_depth=10, # Reduced slightly for initial speed
                min_samples_leaf=5, random_state=42, n_jobs=-1)) # n_jobs=-1 added
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("model",  GradientBoostingRegressor(
                n_estimators=100, learning_rate=0.05,
                max_depth=4, min_samples_leaf=10,
                subsample=0.8, random_state=42))
        ]),
    }

def tune_best_model(X_train, y_train):
    print("\n[4b] Hyperparameter tuning (Gradient Boosting) ...")
    param_grid = {
        "model__n_estimators":  [100, 150],
        "model__learning_rate": [0.05, 0.10],
        "model__max_depth":     [4, 5],
    }
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(subsample=0.8, min_samples_leaf=10, random_state=42))
    ])
    gs = GridSearchCV(pipe, param_grid, cv=3,
                      scoring="neg_root_mean_squared_error",
                      n_jobs=-1, # n_jobs=-1 added
                      verbose=1)  # verbose=1 added for progress
    gs.fit(X_train, y_train)
    print(f"  Best params : {gs.best_params_}")
    print(f"  CV RMSE     : {-gs.best_score_:.2f}")
    return gs.best_estimator_

# 5. VISUALIZATION

def plot_results(results_df, best_name, best_model, X_test, y_test, feature_names, df_raw):
    palette = {"Ridge Regression": "#4C72B0", "Random Forest": "#DD8452", "Gradient Boosting": "#55A868"}
    fig = plt.figure(figsize=(18, 13))
    fig.patch.set_facecolor("#0D1117")
    gs_layout = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)
    text_kw  = dict(color="white")
    spine_kw = "#3A3F4B"

    def style_ax(ax):
        ax.set_facecolor("#161B22")
        for spine in ax.spines.values(): spine.set_color(spine_kw)
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")

    # (a) Model Comparison
    ax1 = fig.add_subplot(gs_layout[0, 0])
    style_ax(ax1)
    models_ = results_df["model"].tolist()
    rmses_  = results_df["RMSE"].tolist()
    colors_ = [palette.get(m, "#888") for m in models_]
    bars = ax1.barh(models_, rmses_, color=colors_, height=0.5)
    for bar, v in zip(bars, rmses_):
        ax1.text(v + 0.3, bar.get_y() + bar.get_height()/2, f"{v:.2f}", va="center", fontsize=9, **text_kw)
    ax1.set_xlabel("RMSE (cycles)", **text_kw)
    ax1.set_title("(a) Model Comparison — RMSE", fontsize=11)
    ax1.invert_yaxis()

    # (b) Pred vs Actual
    ax2 = fig.add_subplot(gs_layout[0, 1])
    style_ax(ax2)
    y_pred = best_model.predict(X_test)
    ax2.scatter(y_test, y_pred, alpha=0.35, s=12, color="#55A868")
    lims = [0, max(y_test.max(), y_pred.max()) + 5]
    ax2.plot(lims, lims, "r--", linewidth=1.2)
    ax2.set_xlabel("Actual RUL (cycles)")
    ax2.set_ylabel("Predicted RUL (cycles)")
    ax2.set_title(f"(b) {best_name} — Predicted vs Actual", fontsize=11)

    # (c) Feature Importance
    ax3 = fig.add_subplot(gs_layout[1, 0])
    style_ax(ax3)
    try:
        importances = best_model.named_steps["model"].feature_importances_
        feat_imp = pd.Series(importances, index=feature_names).nlargest(15)
        feat_imp.sort_values().plot.barh(ax=ax3, color="#4C72B0")
        ax3.set_title("(c) Top-15 Feature Importances", fontsize=11)
    except:
        ax3.text(0.5, 0.5, "N/A", ha="center", va="center", **text_kw)

    # (d) Degradation Profile
    ax4 = fig.add_subplot(gs_layout[1, 1])
    style_ax(ax4)
    sample_unit = df_raw["unit_id"].iloc[0]
    unit_data = df_raw[df_raw["unit_id"] == sample_unit].sort_values("cycle")
    ax4.plot(unit_data["cycle"], unit_data["sensor_09"], color="#55A868", label="Sensor 09")
    ax4b = ax4.twinx()
    ax4b.plot(unit_data["cycle"], unit_data["rul"], color="#F0C040", linestyle="--", label="RUL")
    ax4.set_xlabel("Cycle")
    ax4.set_title(f"(d) Engine #{sample_unit} — Degradation Profile", fontsize=11)

    plt.savefig("rul_prediction_results_optimized.png", dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
# MAIN

def main():
    print("=" * 65)
    print("  Turbofan Engine RUL Prediction | GE Aerospace PM (Optimized)")
    print("=" * 65)

    print("\n[1] Simulating data ...")
    df = simulate_engine_data(n_engines=80, min_life=130, max_life=300)
    df = compute_rul(df, clip_max=130)

    print("[2] Engineering features ...")
    df = engineer_features(df, window=15)
    df = drop_low_variance_sensors(df, threshold=0.01)
    df.dropna(inplace=True)

    print("[3] Splitting data ...")
    train_df, test_df = unit_train_test_split(df, test_frac=0.2)
    drop_cols = ["unit_id", "cycle", "rul"]
    X_train, y_train = train_df.drop(columns=drop_cols), train_df["rul"]
    X_test, y_test   = test_df.drop(columns=drop_cols), test_df["rul"]

    print("\n[4a] Training & evaluating models ...")
    models = build_models()
    results = []
    for name, pipe in models.items():
        print(f"  Training {name} ...") # Added visibility
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        results.append(evaluate(name, y_test.values, y_pred))

    results_df = pd.DataFrame(results).sort_values("RMSE")
    
    # Only tune if model names match logic
    best_model = tune_best_model(X_train, y_train)
    y_pred_tuned = best_model.predict(X_test)
    evaluate("GB (Tuned)", y_test.values, y_pred_tuned)

    print("\n[5] Generating visualization ...")
    plot_results(results_df, "Gradient Boosting (Tuned)", best_model, X_test, y_test, X_train.columns.tolist(), df)

    print("\n" + "=" * 65)
    print("  DONE. Result saved as 'rul_prediction_results_optimized.png'")
    print("=" * 65)

if __name__ == "__main__":
    main()
