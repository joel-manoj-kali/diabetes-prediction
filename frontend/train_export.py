import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

FEATURES = ['num_preg', 'glucose_conc', 'diastolic_bp', 'insulin',
            'bmi', 'diab_pred', 'age', 'skin']
ZERO_MISSING = ['glucose_conc', 'diastolic_bp', 'insulin', 'bmi', 'skin']

df = pd.read_csv('repo/data/pima-data.csv')
df['diabetes'] = df['diabetes'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})

X = df[FEATURES].astype(float).values
y = df['diabetes'].values.astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=10, stratify=y)

# zero -> train-mean imputation (as in the notebook, but fit on train only)
means = {}
for f in ZERO_MISSING:
    i = FEATURES.index(f)
    col = X_train[:, i]
    m = float(col[col != 0].mean())
    means[f] = m
    X_train[X_train[:, i] == 0, i] = m
    X_test[X_test[:, i] == 0, i] = m

clf = RandomForestClassifier(n_estimators=120, max_depth=5, min_samples_leaf=6,
                             max_features='sqrt', random_state=10, n_jobs=-1)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
proba = clf.predict_proba(X_test)[:, 1]
acc = metrics.accuracy_score(y_test, pred)
auc = metrics.roc_auc_score(y_test, proba)
rec = metrics.recall_score(y_test, pred)
prec = metrics.precision_score(y_test, pred)
cv = cross_val_score(clf, X, y, cv=5, scoring='accuracy').mean()
print(f"acc={acc:.4f} auc={auc:.4f} recall={rec:.4f} prec={prec:.4f} cv={cv:.4f}")
print("importances:", dict(zip(FEATURES, clf.feature_importances_.round(4))))

# export trees compactly: arrays of [feature, threshold, left, right, p1]
trees = []
for est in clf.estimators_:
    t = est.tree_
    nodes = []
    for i in range(t.node_count):
        if t.children_left[i] == -1:
            v = t.value[i][0]
            # sklearn >=1.4 stores normalized class proportions for RF
            p = float(v[1] / v.sum()) if v.sum() > 0 else 0.0
            nodes.append([-1, 0, -1, -1, round(p, 5)])
        else:
            nodes.append([int(t.feature[i]), round(float(t.threshold[i]), 6),
                          int(t.children_left[i]), int(t.children_right[i]), 0])
    trees.append(nodes)

# dataset distributions for the "how you compare" view
dist = {}
clean = df[FEATURES].astype(float).copy()
for f in ZERO_MISSING:
    clean.loc[clean[f] == 0, f] = means[f]
for f in FEATURES:
    s = clean[f]
    dist[f] = {
        'min': float(s.min()), 'max': float(s.max()),
        'p25': float(s.quantile(.25)), 'p50': float(s.median()),
        'p75': float(s.quantile(.75)),
        'mean_neg': float(clean.loc[y == 0, f].mean()),
        'mean_pos': float(clean.loc[y == 1, f].mean()),
    }

out = {
    'features': FEATURES,
    'means': means,
    'trees': trees,
    'metrics': {'accuracy': round(acc, 4), 'auc': round(auc, 4),
                'recall': round(rec, 4), 'precision': round(prec, 4),
                'cv_accuracy': round(cv, 4),
                'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
                'n_total': int(len(y)), 'pos_rate': round(float(y.mean()), 4),
                'n_trees': len(trees), 'max_depth': 5},
    'importances': {f: round(float(v), 4)
                    for f, v in zip(FEATURES, clf.feature_importances_)},
    'dist': dist,
}
with open('model.json', 'w') as fh:
    json.dump(out, fh, separators=(',', ':'))
import os
print("model.json size:", os.path.getsize('model.json'))

# sanity check: JS-equivalent scoring in python
def score(x):
    tot = 0.0
    for nodes in trees:
        i = 0
        while nodes[i][0] != -1:
            f, thr, l, r, _ = nodes[i]
            i = l if x[f] <= thr else r
        tot += nodes[i][4]
    return tot / len(trees)

manual = np.array([score(row) for row in X_test])
print("max diff vs sklearn:", np.abs(manual - proba).max())
