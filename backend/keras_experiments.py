"""
SafeRoute: Keras experiments for the paper revision.
Run from the project's backend folder:
    python keras_experiments.py --selftest            # checks data prep, no TensorFlow needed
    python keras_experiments.py --part importance     # Table III weights from the trained model (fast)
    python keras_experiments.py --part ablation --seeds 3 --epochs 250   # with vs without attention
    python keras_experiments.py --part lodo --epochs 250                 # leave-district-out, proposed model
    python keras_experiments.py --part all
Results are saved to keras_results.json
"""
import argparse, json, os, warnings
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

DATA   = "datasets/training/route_safety_dataset_cleaned.csv"
MODEL  = "models/safe_route_final_multibranch.keras"
SCALER = "models/safe_route_final_scalers.pkl"
GROUPS = {
 "traffic":  ["traffic_index","traffic_safety_score"],
 "lighting": ["lighting_score"],
 "weather":  ["rainfall_mm","rainfall_risk","visibility_km","visibility_safety_score","month_sin","month_cos"],
 "crime":    ["violent_crime_count","property_crime_count","women_crime_count","other_crime_count",
              "total_crime_count","crime_rate_per_1000_exposure","crime_intensity","crime_pressure"],
 "emergency":["nearest_police_km","nearest_hospital_km","emergency_access_score","emergency_distance"],
}
ORDER = list(GROUPS)
LABEL = {"traffic":"beta (traffic)","lighting":"delta (lighting)","weather":"gamma (weather)","crime":"alpha (crime)","emergency":"epsilon (emergency)"}

def load_data():
    d = pd.read_csv(DATA, parse_dates=["date"]).dropna().reset_index(drop=True)
    d["safety_score"] = d.safety_score.clip(0, 100)
    d["traffic_safety_score"] = d.traffic_index.clip(0, 100)
    d["rainfall_risk"] = (d.rainfall_mm / max(d.rainfall_mm.max(), 1.0) * 100).clip(0, 100)
    vmin, vmax = d.visibility_km.min(), d.visibility_km.max()
    d["visibility_safety_score"] = ((d.visibility_km - vmin) / (vmax - vmin) * 100).clip(0, 100)
    d["crime_intensity"] = d.total_crime_count / max(d.total_crime_count.median(), 1.0)
    d["crime_pressure"] = (.25*d.violent_crime_count + .20*d.property_crime_count + .20*d.women_crime_count
                           + .15*d.other_crime_count + .20*d.crime_rate_per_1000_exposure)
    d["emergency_distance"] = d.nearest_police_km + d.nearest_hospital_km
    m = d.date.dt.month
    d["month_sin"] = np.sin(2*np.pi*m/12); d["month_cos"] = np.cos(2*np.pi*m/12)
    return d

def split(d):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(d, groups=d.route_id))
    return tr, te

def arrays(d, idx):
    return {g: d.loc[idx, GROUPS[g]].values.astype("float32") for g in ORDER}

def fit_scalers(d, idx):
    return {g: StandardScaler().fit(d.loc[idx, GROUPS[g]].values) for g in ORDER}

def apply_scalers(sc, arr):
    return {g: sc[g].transform(arr[g]).astype("float32") for g in ORDER}

def cls(s): return np.digitize(s, [30, 45, 60, 75])

def metrics(y, p):
    p = np.clip(np.asarray(p).ravel(), 0, 100); e = y - p
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e**2).mean())),
                R2=float(1 - (e**2).sum()/((y-y.mean())**2).sum()),
                Closeness=float(max(0, 100*(1-np.abs(e).mean()/np.abs(y).mean()))),
                Acc5=float(100*(cls(y) == cls(p)).mean()))

# ---------------------------------------------------------------- TensorFlow parts
def tf_imports():
    import tensorflow as tf
    return tf

def make_scale_layer(tf):
    @tf.keras.utils.register_keras_serializable(package="SafeRoute", name="ScaleTo100")
    class ScaleTo100(tf.keras.layers.Layer):
        def call(self, x): return x * 100.0
    return ScaleTo100

def build_model(tf, attention=True, seed=42):
    from tensorflow.keras import layers as L, Model, Input
    tf.keras.utils.set_random_seed(seed)
    ScaleTo100 = make_scale_layer(tf)
    def branch(x):
        x = L.Dense(64, activation="relu")(x); x = L.BatchNormalization()(x)
        x = L.Dense(64, activation="relu")(x); x = L.Dropout(0.15)(x)
        return L.Dense(32, activation="relu")(x)
    ins, toks = [], []
    for g in ORDER:
        i = Input(shape=(len(GROUPS[g]),), name=f"{g}_input"); ins.append(i)
        toks.append(L.Reshape((1, 32))(branch(i)))
    t = L.Concatenate(axis=1)(toks)
    if attention:
        a = L.MultiHeadAttention(num_heads=4, key_dim=8, dropout=0.10)(t, t, t)
        t = L.LayerNormalization()(L.Add()([t, a]))
        f = L.Dense(64, activation="relu")(t); f = L.Dropout(0.15)(f); f = L.Dense(32)(f)
        t = L.LayerNormalization()(L.Add()([t, f]))
    x = L.GlobalAveragePooling1D()(t)
    x = L.Dense(128, activation="relu")(x); x = L.BatchNormalization()(x); x = L.Dropout(0.20)(x)
    x = L.Dense(64, activation="relu")(x);  x = L.BatchNormalization()(x); x = L.Dropout(0.15)(x)
    x = L.Dense(32, activation="relu")(x);  x = L.Dense(16, activation="relu")(x)
    out = ScaleTo100()(L.Dense(1, activation="sigmoid")(x))
    m = Model(ins, out)
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=tf.keras.losses.Huber())
    return m

def train_eval(tf, d, tr, te, attention, seed, epochs):
    sc = fit_scalers(d, tr)
    Xtr = apply_scalers(sc, arrays(d, tr)); Xte = apply_scalers(sc, arrays(d, te))
    ytr, yte = d.loc[tr, "safety_score"].values.astype("float32"), d.loc[te, "safety_score"].values
    m = build_model(tf, attention, seed)
    cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
          tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)]
    m.fit([Xtr[g] for g in ORDER], ytr, validation_split=0.15, epochs=epochs, batch_size=64, callbacks=cb, verbose=0)
    return metrics(yte, m.predict([Xte[g] for g in ORDER], verbose=0))

def agg(rs): return {k: [float(np.mean([r[k] for r in rs])), float(np.std([r[k] for r in rs]))] for k in rs[0]}

def part_importance(tf, d):
    ScaleTo100 = make_scale_layer(tf)
    model = tf.keras.models.load_model(MODEL, custom_objects={"ScaleTo100": ScaleTo100, "SafeRoute>ScaleTo100": ScaleTo100}, compile=False)
    sc = joblib.load(SCALER)
    tr, te = split(d); Xte = apply_scalers(sc, arrays(d, te)); y = d.loc[te, "safety_score"].values
    pred = lambda X: model.predict([X[g] for g in ORDER], verbose=0)
    base = metrics(y, pred(Xte)); print("Trained model on test routes:", {k: round(v, 4) for k, v in base.items()})
    rng = np.random.default_rng(0); drops = {}
    for g in ORDER:
        r = []
        for _ in range(5):
            X = {k: v.copy() for k, v in Xte.items()}; X[g] = X[g][rng.permutation(len(te))]
            r.append(base["R2"] - metrics(y, pred(X))["R2"])
        drops[g] = float(np.mean(r))
    pos = {g: max(v, 0) for g, v in drops.items()}; s = sum(pos.values())
    w = {g: pos[g]/s for g in ORDER}
    print("R2 drop per group:", {g: round(v, 4) for g, v in drops.items()})
    for g in ORDER: print(f"  {LABEL[g]:22s} {w[g]:.3f}")
    return dict(test_metrics=base, r2_drop=drops, weights=w)

def part_ablation(tf, d, seeds, epochs):
    tr, te = split(d); out = {}
    for name, att in [("proposed (attention)", True), ("without attention", False)]:
        rs = []
        for s in range(seeds):
            r = train_eval(tf, d, tr, te, att, 42+s, epochs); rs.append(r); print(name, "seed", 42+s, {k: round(v, 3) for k, v in r.items()}, flush=True)
        out[name] = agg(rs)
    return out

def part_lodo(tf, d, epochs):
    rs = []
    for k, (a, b) in enumerate(GroupKFold(5).split(d, groups=d.district.values)):
        r = train_eval(tf, d, a, b, True, 42, epochs); rs.append(r); print("fold", k, {kk: round(v, 3) for kk, v in r.items()}, flush=True)
    return agg(rs)

def selftest(d):
    tr, te = split(d)
    print("rows", len(d), "| train/test samples", len(tr), len(te), "| train/test routes", d.route_id.iloc[tr].nunique(), d.route_id.iloc[te].nunique())
    sc = joblib.load(SCALER); ok = True
    for g in ORDER:
        mine = d.loc[tr, GROUPS[g]].values.mean(0); saved = sc[g].mean_
        diff = np.abs(mine - saved) / (np.abs(saved) + 1e-9)
        print(f"  {g:10s} max relative diff between my feature means and the saved scaler means: {diff.max():.4f}")
        ok &= bool(diff.max() < 0.02)
    print("Feature engineering matches the saved scalers:", "YES" if ok else "NO - check the feature definitions")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="importance", choices=["importance", "ablation", "lodo", "all"])
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(); d = load_data()
    if a.selftest: selftest(d); raise SystemExit
    tf = tf_imports(); res = {}
    if a.part in ("importance", "all"): res["importance"] = part_importance(tf, d)
    if a.part in ("ablation", "all"):   res["ablation"] = part_ablation(tf, d, a.seeds, a.epochs)
    if a.part in ("lodo", "all"):       res["lodo_proposed"] = part_lodo(tf, d, a.epochs)
    json.dump(res, open("keras_results.json", "w"), indent=1); print("saved keras_results.json")
