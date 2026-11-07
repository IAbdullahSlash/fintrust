"""FI-2010 (NoAuction, Z-score) forecasts for FinTrust replay.
Binary task: y=1 if the k=10 mid-price label is 'up' (label 1), else 0.
Days 1-5 train, day 6 calibrate (Platt), days 7-10 out-of-sample forecasts.
Input: folder of day01.npy..day10.npy (149 x N arrays converted from the original .txt files).
"""
import sys, numpy as np, json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
SRC = Path(sys.argv[1]); OUT = Path(__file__).resolve().parent
days = {d: np.load(SRC / f'day{d:02d}.npy') for d in range(1, 11)}
X = {d: days[d][:144].T.astype(np.float32) for d in days}
Y = {d: (days[d][148] == 1).astype(int) for d in days}
H = 10
def purge(d):  # drop last H samples of a day so labels do not cross the day boundary
    return X[d][:-H], Y[d][:-H]
Xtr = np.vstack([purge(d)[0] for d in range(1, 6)]); ytr = np.concatenate([purge(d)[1] for d in range(1, 6)])
Xca, yca = purge(6)
rng = np.random.default_rng(0); sub = rng.choice(len(ytr), size=min(80000, len(ytr)), replace=False)
models = {
    'LOGIT40': (LogisticRegression(max_iter=2000, C=0.1), slice(0, 40)),
    'GBT': (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=0), slice(0, 144)),
    'MLP': (MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=60, early_stopping=True, random_state=0), slice(0, 144)),
}
test_days = [7, 8, 9, 10]
Xte = np.vstack([X[d] for d in test_days]); yte = np.concatenate([Y[d] for d in test_days])
day_id = np.concatenate([[d] * len(Y[d]) for d in test_days])
P = {}; info = {}
for name, (m, cols) in models.items():
    xs = Xtr[sub] if name != 'LOGIT40' else Xtr
    ys = ytr[sub] if name != 'LOGIT40' else ytr
    m.fit(xs[:, cols], ys)
    logit = lambda Z: np.log(np.clip(m.predict_proba(Z[:, cols])[:, 1], 1e-6, 1 - 1e-6) / (1 - np.clip(m.predict_proba(Z[:, cols])[:, 1], 1e-6, 1 - 1e-6)))
    platt = LogisticRegression().fit(logit(Xca).reshape(-1, 1), yca)
    P[name] = platt.predict_proba(logit(Xte).reshape(-1, 1))[:, 1]
    info[name] = dict(brier_cal_day6=float(np.mean((platt.predict_proba(logit(Xca).reshape(-1, 1))[:, 1] - yca) ** 2)),
                      brier_days9_10=float(np.mean((P[name][day_id >= 9] - yte[day_id >= 9]) ** 2)))
    print(name, info[name], flush=True)
info['base_rate_train'] = float(ytr.mean()); info['base_rate_test'] = float(yte[day_id >= 9].mean())
info['n_train'] = int(len(ytr)); info['n_test_days9_10'] = int((day_id >= 9).sum()); info['n_replay'] = int(len(yte))
np.savez_compressed(OUT / 'fi2010_forecasts.npz', y=yte, day=day_id, **P)
(OUT / 'fi2010_agents.json').write_text(json.dumps(info, indent=1)); print(info)
