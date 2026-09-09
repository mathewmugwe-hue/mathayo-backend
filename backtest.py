"""
Real backtest: pre-match-only features, walk-forward validation, top-5
European leagues, 2010-2026 seasons. No in-match/post-match statistics
used anywhere (no shots, corners, cards, HT score) - those don't exist
before kickoff and would be data leakage for a betting model.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

df = pd.read_csv('matches_full.csv', low_memory=False)

TOP5 = ['E0', 'D1', 'SP1', 'I1', 'F1']
df = df[df['Division'].isin(TOP5)].copy()
df['MatchDate'] = pd.to_datetime(df['MatchDate'], errors='coerce')
df = df.dropna(subset=['MatchDate'])
df = df[df['MatchDate'] >= '2010-01-01']

required = ['MatchDate', 'Division', 'HomeTeam', 'AwayTeam',
            'HomeElo', 'AwayElo', 'Form3Home', 'Form5Home', 'Form3Away', 'Form5Away',
            'OddHome', 'OddDraw', 'OddAway', 'FTResult']
df = df[required].dropna()
df = df.sort_values('MatchDate').reset_index(drop=True)

print(f"Rows after cleaning (pre-match features only, 2010+, top-5 leagues): {len(df)}")

df['EloDiff'] = df['HomeElo'] - df['AwayElo']
df['EloTotal'] = df['HomeElo'] + df['AwayElo']
df['Form5Diff'] = df['Form5Home'] - df['Form5Away']
df['Form3Diff'] = df['Form3Home'] - df['Form3Away']

inv_h = 1 / df['OddHome']
inv_d = 1 / df['OddDraw']
inv_a = 1 / df['OddAway']
overround = inv_h + inv_d + inv_a
df['MktProbHome'] = inv_h / overround
df['MktProbDraw'] = inv_d / overround
df['MktProbAway'] = inv_a / overround

feature_cols = ['EloDiff', 'EloTotal', 'Form5Diff', 'Form3Diff',
                 'Form5Home', 'Form5Away', 'MktProbHome', 'MktProbDraw', 'MktProbAway']

X = df[feature_cols].values
y = df['FTResult'].values

n = len(df)
test_start = int(n * 0.75)
fold_size = 1000
folds = []
i = test_start
while i + fold_size <= n:
    folds.append((i, i + fold_size))
    i += fold_size
if not folds:
    folds = [(test_start, n)]

model_correct, market_correct, total = 0, 0, 0
model_probs_all, market_probs_all, y_true_all = [], [], []
label_map = {'H': 0, 'D': 1, 'A': 2}

for start, end in folds:
    X_train, y_train = X[:start], y[:start]
    X_test, y_test = X[start:end], y[start:end]

    clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05, random_state=42)
    y_train_enc = np.array([label_map[v] for v in y_train])
    clf.fit(X_train, y_train_enc)

    y_test_enc = np.array([label_map[v] for v in y_test])
    proba = clf.predict_proba(X_test)
    pred = proba.argmax(axis=1)
    model_correct += (pred == y_test_enc).sum()

    mkt_proba = df.iloc[start:end][['MktProbHome', 'MktProbDraw', 'MktProbAway']].values
    mkt_pred = mkt_proba.argmax(axis=1)
    market_correct += (mkt_pred == y_test_enc).sum()

    total += len(y_test)
    model_probs_all.append(proba)
    market_probs_all.append(mkt_proba)
    y_true_all.append(y_test_enc)

model_probs_all = np.vstack(model_probs_all)
market_probs_all = np.vstack(market_probs_all)
y_true_all = np.concatenate(y_true_all)

model_acc = model_correct / total
market_acc = market_correct / total
model_logloss = log_loss(y_true_all, model_probs_all, labels=[0, 1, 2])
market_logloss = log_loss(y_true_all, market_probs_all, labels=[0, 1, 2])

def rps(probs, true_idx):
    n_classes = probs.shape[1]
    cum_probs = np.cumsum(probs, axis=1)
    true_onehot = np.zeros_like(probs)
    true_onehot[np.arange(len(true_idx)), true_idx] = 1
    cum_true = np.cumsum(true_onehot, axis=1)
    return np.mean(np.sum((cum_probs - cum_true) ** 2, axis=1) / (n_classes - 1))

model_rps = rps(model_probs_all, y_true_all)
market_rps = rps(market_probs_all, y_true_all)

last_idx = min(i, n - 1)
print()
print(f"Test set size: {total} matches (walk-forward, held-out future data only)")
print(f"Date range tested: {df.iloc[test_start]['MatchDate'].date()} to {df.iloc[last_idx]['MatchDate'].date()}")
print()
print(f"{'Metric':<20} {'Trained Model':<18} {'Market Favorite':<18}")
print(f"{'Accuracy':<20} {model_acc*100:.2f}%{'':<11} {market_acc*100:.2f}%")
print(f"{'Log Loss':<20} {model_logloss:.4f}{'':<12} {market_logloss:.4f}")
print(f"{'RPS':<20} {model_rps:.4f}{'':<12} {market_rps:.4f}")
print()
print(f"For a 15-match slip: {model_acc*100:.1f}% accuracy implies an EXPECTED "
      f"{model_acc*15:.1f} correct picks out of 15.")
