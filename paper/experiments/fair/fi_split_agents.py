"""Train the three FI-2010 agents for one anchored split and save out-of-sample forecasts.
usage: python fi_split_agents.py <dayNN.npy folder> <name> <last_train_day> <cal_day> <first_day> <last_day>
Same models and settings as ../real/fi2010_agents.py."""
import sys, json, numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
SRC = Path(sys.argv[1]); name = sys.argv[2]; tr_end, cal, d0, d1 = map(int, sys.argv[3:7]); OUT = Path(__file__).resolve().parent
H = 10
days = {d: np.load(SRC / f'day{d:02d}.npy') for d in range(1, 11)}
X = {d: days[d][:144].T.astype(np.float32) for d in days}; Y = {d: (days[d][148] == 1).astype(int) for d in days}
Xtr = np.vstack([X[d][:-H] for d in range(1, tr_end + 1)]); ytr = np.concatenate([Y[d][:-H] for d in range(1, tr_end + 1)])
Xca, yca = X[cal][:-H], Y[cal][:-H]
rng = np.random.default_rng(0); sub = rng.choice(len(ytr), size=min(80000, len(ytr)), replace=False)
models = {'LOGIT40': (LogisticRegression(max_iter=2000, C=0.1), slice(0, 40)),
          'GBT': (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=31, random_state=0), slice(0, 144)),
          'MLP': (MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, max_iter=60, early_stopping=True, random_state=0), slice(0, 144))}
fd = list(range(d0, d1 + 1)); Xte = np.vstack([X[d] for d in fd]); yte = np.concatenate([Y[d] for d in fd]); day = np.concatenate([[d] * len(Y[d]) for d in fd])
P = {}
for nm, (m, cols) in models.items():
    xs, ys = (Xtr, ytr) if nm == 'LOGIT40' else (Xtr[sub], ytr[sub])
    m.fit(xs[:, cols], ys)
    def lg(Z):
        q = np.clip(m.predict_proba(Z[:, cols])[:, 1], 1e-6, 1 - 1e-6); return np.log(q / (1 - q))
    pl = LogisticRegression().fit(lg(Xca).reshape(-1, 1), yca)
    P[nm] = pl.predict_proba(lg(Xte).reshape(-1, 1))[:, 1]
np.savez_compressed(OUT / f'fi_{name}_forecasts.npz', y=yte, day=day, **P)
print(name, {k: round(float(np.mean((v - yte) ** 2)), 4) for k, v in P.items()}, len(yte))
