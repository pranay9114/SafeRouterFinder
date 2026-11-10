"""
Leave-district-out evaluation of the proposed network (answers reviewer comment 7).
Run from the project's backend folder:   python run_lodo.py
Optional: python run_lodo.py --epochs 100     (faster, but state it in the paper)
"""
import argparse, json, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")

DATA = "datasets/training/route_safety_dataset_cleaned.csv"
GROUPS = {
 "traffic":  ["traffic_index","traffic_safety_score"],
 "lighting": ["lighting_score"],
 "weather":  ["rainfall_mm","rainfall_risk","visibility_km","visibility_safety_score","month_sin","month_cos"],
 "crime":    ["violent_crime_count","property_crime_count","women_crime_count","other_crime_count",
              "total_crime_count","crime_rate_per_1000_exposure","crime_intensity","crime_pressure"],
 "emergency":["nearest_police_km","nearest_hospital_km","emergency_access_score","emergency_distance"],
}
ORDER = list(GROUPS)

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

cls = lambda s: np.digitize(s, [30, 45, 60, 75])

def metrics(y, p):
    p = np.clip(np.asarray(p).ravel(), 0, 100); e = y - p
    return dict(MAE=float(np.abs(e).mean()), RMSE=float(np.sqrt((e**2).mean())),
                R2=float(1 - (e**2).sum()/((y-y.mean())**2).sum()),
                Closeness=float(max(0, 100*(1-np.abs(e).mean()/np.abs(y).mean()))),
                Acc5=float(100*(cls(y) == cls(p)).mean()))

def build_model(tf, seed=42):
    from tensorflow.keras import layers as L, Model, Input
    tf.keras.utils.set_random_seed(seed)
    class ScaleTo100(L.Layer):
        def call(self, x): return x * 100.0
    def branch(x):
        x = L.Dense(64, activation="relu")(x); x = L.BatchNormalization()(x)
        x = L.Dense(64, activation="relu")(x); x = L.Dropout(0.15)(x)
        return L.Dense(32, activation="relu")(x)
    ins, toks = [], []
    for g in ORDER:
        i = Input(shape=(len(GROUPS[g]),), name=f"{g}_input"); ins.append(i)
        toks.append(L.Reshape((1, 32))(branch(i)))
    t = L.Concatenate(axis=1)(toks)
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

def main(epochs):
    import tensorflow as tf
    d = load_data(); folds = []
    for k, (a, b) in enumerate(GroupKFold(5).split(d, groups=d.district.values)):
        t0 = time.time()
        held = sorted(d.district.iloc[b].unique())
        sc = {g: StandardScaler().fit(d.loc[a, GROUPS[g]].values) for g in ORDER}
        Xa = [sc[g].transform(d.loc[a, GROUPS[g]].values).astype("float32") for g in ORDER]
        Xb = [sc[g].transform(d.loc[b, GROUPS[g]].values).astype("float32") for g in ORDER]
        ya, yb = d.loc[a, "safety_score"].values.astype("float32"), d.loc[b, "safety_score"].values
        model = build_model(tf)
        cb = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True),
              tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-6)]
        model.fit(Xa, ya, validation_split=0.15, epochs=epochs, batch_size=64, callbacks=cb, verbose=0)
        r = metrics(yb, model.predict(Xb, verbose=0)); r["held_out_districts"] = held; folds.append(r)
        print(f"fold {k}: " + ", ".join(f"{m}={r[m]:.3f}" for m in ["MAE","RMSE","R2","Closeness","Acc5"]) + f"  ({time.time()-t0:.0f} s)", flush=True)
        print("   held-out districts:", ", ".join(held), flush=True)
        json.dump(folds, open("lodo_results.json", "w"), indent=1)
    print("\n=== Leave-district-out, proposed network (mean +/- std over 5 folds) ===")
    for m in ["MAE","RMSE","R2","Closeness","Acc5"]:
        v = [f[m] for f in folds]; print(f"{m:10s} {np.mean(v):.3f} +/- {np.std(v):.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=250)
    main(ap.parse_args().epochs)
