"""IN6227 Variant 1: reproducible classification comparison."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import warnings

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).parent / '.mplconfig'))
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    RocCurveDisplay, PrecisionRecallDisplay, classification_report)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

SEED = 42
TARGET = 'label'


def save_json(value, path):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
        default=lambda v: v.item() if isinstance(v, np.generic) else str(v)), encoding='utf-8')


def load_data(path):
    frame = pd.read_csv(path)
    if TARGET not in frame:
        raise ValueError(f'{path}: missing label column')
    # Whitespace cleaning is deterministic; no distribution is learned here.
    for col in frame.select_dtypes(exclude='number'):
        frame[col] = frame[col].astype('string').str.strip().replace('', pd.NA)
        frame[col] = frame[col].astype(object).where(frame[col].notna(), np.nan)
    frame = frame.replace([np.inf, -np.inf], np.nan)
    valid_labels = set(frame[TARGET].dropna())
    if not valid_labels <= {'yes', 'no'}:
        raise ValueError(f'Unexpected labels: {valid_labels}')
    return frame


def describe(frame):
    return {'rows': len(frame), 'columns': len(frame.columns),
        'missing_by_column': frame.isna().sum().to_dict(),
        'class_counts': frame[TARGET].value_counts().to_dict(),
        'exact_duplicate_rows': int(frame.duplicated().sum()),
        'unique_by_column': frame.nunique().to_dict()}


def preprocessor(numeric, categorical, scale):
    num_steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if scale:
        num_steps.append(('scale', StandardScaler()))
    return ColumnTransformer([
        ('numeric', Pipeline(num_steps), numeric),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent', keep_empty_features=True)),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ]), categorical)
    ])


def evaluate(model, X, y):
    prediction = model.predict(X)
    positive_column = list(model.classes_).index(1)
    probability = model.predict_proba(X)[:, positive_column]
    return {'accuracy': accuracy_score(y, prediction),
        'balanced_accuracy': balanced_accuracy_score(y, prediction),
        'precision_yes': precision_score(y, prediction, zero_division=0),
        'recall_yes': recall_score(y, prediction, zero_division=0),
        'f1_yes': f1_score(y, prediction, zero_division=0),
        'roc_auc': roc_auc_score(y, probability),
        'average_precision': average_precision_score(y, probability)}, prediction, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('D:/dataset'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs')
    parser.add_argument('--jobs', type=int, default=1)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings('error', category=ConvergenceWarning)

    raw_train = load_data(args.data_dir / 'train.csv')
    audit = {'train': describe(raw_train)}
    # Remove exact duplicates before CV so copies cannot cross fold boundaries.
    train = raw_train.dropna(subset=[TARGET]).drop_duplicates().copy()
    X = train.drop(columns=TARGET)
    y = train[TARGET].map({'no': 0, 'yes': 1}).astype(int)
    numeric = X.select_dtypes(include='number').columns.tolist()
    categorical = [c for c in X if c not in numeric]
    audit.update(numeric_features=numeric, categorical_features=categorical,
        usable_train_rows=len(train), positive_label='yes', seed=SEED)
    X.describe(include='all').to_csv(out / 'train_description.csv')
    # IQR flags are diagnostic only: without domain evidence, extremes are retained.
    q1, q3 = X[numeric].quantile(.25), X[numeric].quantile(.75)
    iqr = q3 - q1
    audit['train_iqr_outlier_counts'] = ((X[numeric] < q1 - 1.5 * iqr) |
        (X[numeric] > q3 + 1.5 * iqr)).sum().to_dict()
    audit['conflicting_feature_groups'] = int((train.groupby(list(X.columns), dropna=False)[TARGET].nunique() > 1).sum())
    fig, axes = plt.subplots(2, 4, figsize=(13, 6))
    for col, ax in zip(numeric, axes.flat):
        X[col].plot.hist(bins=40, ax=ax, title=col, color='#287c8e')
    axes.flat[-1].bar(['no', 'yes'], [int((y == 0).sum()), int((y == 1).sum())])
    axes.flat[-1].set_title('Training class counts')
    fig.tight_layout(); fig.savefig(out / 'data_exploration.png', dpi=160); plt.close(fig)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    configurations = {
        'logistic_regression': (
            LogisticRegression(solver='lbfgs', max_iter=3000, tol=1e-4, random_state=SEED),
            {'model__C': [.01, .1, 1, 10], 'model__class_weight': [None, 'balanced']}, True),
        'decision_tree': (
            DecisionTreeClassifier(criterion='gini', random_state=SEED),
            {'model__max_depth': [3, 5, 8, 12, None],
             'model__min_samples_leaf': [5, 20, 50],
             'model__class_weight': [None, 'balanced']}, False)
    }
    fitted, tuning = {}, {}
    for name, (classifier, grid, scale) in configurations.items():
        print(f'Tuning {name} with training-only 5-fold CV...', flush=True)
        pipeline = Pipeline([('preprocess', preprocessor(numeric, categorical, scale)), ('model', classifier)])
        search = GridSearchCV(pipeline, grid, scoring={'f1': 'f1', 'roc_auc': 'roc_auc',
            'balanced_accuracy': 'balanced_accuracy'}, refit='f1', cv=cv,
            n_jobs=args.jobs, return_train_score=True, error_score='raise')
        search.fit(X, y)
        pd.DataFrame(search.cv_results_).to_csv(out / f'{name}_cv_results.csv', index=False)
        fitted[name] = search.best_estimator_
        tuning[name] = {'best_parameters': search.best_params_, 'cv_f1_mean': search.best_score_,
            'cv_f1_std': search.cv_results_['std_test_f1'][search.best_index_]}
        classifier = fitted[name].named_steps['model']
        tuning[name]['fitted_size'] = ({'iterations': classifier.n_iter_.tolist()} if scale else
            {'depth': classifier.get_depth(), 'leaves': classifier.get_n_leaves()})
        joblib.dump(fitted[name], out / f'{name}.joblib')
        names = fitted[name].named_steps['preprocess'].get_feature_names_out()
        weights = classifier.coef_[0] if scale else classifier.feature_importances_
        pd.DataFrame({'feature': names, 'coefficient' if scale else 'importance': weights}).to_csv(
            out / f'{name}_features.csv', index=False)
    selected = max(tuning, key=lambda name: tuning[name]['cv_f1_mean'])
    save_json({'selected_by_cv': selected, 'models': tuning}, out / 'tuning.json')

    # Only after all model selection is finished do we load/evaluate the test set.
    raw_test = load_data(args.data_dir / 'test.csv')
    if set(raw_test.columns) != set(raw_train.columns):
        raise ValueError('Train/test schemas differ')
    audit['test'] = describe(raw_test)
    test = raw_test.dropna(subset=[TARGET]).copy()
    Xt = test[X.columns]
    yt = test[TARGET].map({'no': 0, 'yes': 1}).astype(int)
    if yt.nunique() != 2:
        raise ValueError('Test evaluation requires both classes')
    train_keys = pd.MultiIndex.from_frame(X)
    audit['test_feature_overlap_rows'] = int(pd.MultiIndex.from_frame(Xt).isin(train_keys).sum())
    audit['usable_test_rows'] = len(test)
    audit['unseen_test_categories'] = {c: sorted(set(Xt[c].dropna()) - set(X[c].dropna())) for c in categorical}
    save_json(audit, out / 'data_audit.json')
    if audit['test_feature_overlap_rows']:
        print('WARNING: train/test feature overlap detected; see data_audit.json', flush=True)

    rows = []
    predictions = pd.DataFrame({'csv_row_number': test.index + 2, 'actual': test[TARGET]})
    baseline = DummyClassifier(strategy='most_frequent').fit(X, y)
    models = {'majority_baseline': baseline, **fitted}
    fig_cm, cm_axes = plt.subplots(1, 3, figsize=(12, 3.5))
    fig_curves, curve_axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, model) in zip(cm_axes, models.items()):
        metrics, pred, prob = evaluate(model, Xt, yt)
        training_metrics, _, _ = evaluate(model, X, y)
        rows.append({'model': name, 'train_f1_yes': training_metrics['f1_yes'],
            'cv_f1_mean': tuning.get(name, {}).get('cv_f1_mean', np.nan),
            'cv_f1_std': tuning.get(name, {}).get('cv_f1_std', np.nan), **metrics})
        predictions[f'{name}_prediction'] = np.where(pred == 1, 'yes', 'no')
        predictions[f'{name}_probability_yes'] = prob
        save_json(classification_report(yt, pred, target_names=['no', 'yes'], output_dict=True,
            zero_division=0), out / f'{name}_classification_report.json')
        ConfusionMatrixDisplay(confusion_matrix(yt, pred, labels=[0, 1]), display_labels=['no', 'yes']).plot(ax=ax, colorbar=False, values_format='d')
        ax.set_title(name.replace('_', ' ').title(), fontsize=10)
        RocCurveDisplay.from_predictions(yt, prob, name=name, ax=curve_axes[0])
        PrecisionRecallDisplay.from_predictions(yt, prob, name=name, ax=curve_axes[1])
    for fig, filename in [(fig_cm, 'confusion_matrices.png'), (fig_curves, 'roc_pr_curves.png')]:
        fig.tight_layout(); fig.savefig(out / filename, dpi=160); plt.close(fig)
    pd.DataFrame(rows).to_csv(out / 'model_comparison.csv', index=False)
    predictions.to_csv(out / 'test_predictions.csv', index=False)
    tree = fitted['decision_tree']
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(tree.named_steps['model'], max_depth=2, filled=True, fontsize=8,
        feature_names=tree.named_steps['preprocess'].get_feature_names_out(), class_names=['no', 'yes'], ax=ax)
    ax.set_title('Decision tree: first three levels (deeper branches omitted)')
    fig.tight_layout(); fig.savefig(out / 'tree_preview.png', dpi=160); plt.close(fig)
    save_json({'python': platform.python_version(), 'sklearn': sklearn.__version__,
        'pandas': pd.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__,
        'data_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [args.data_dir / 'train.csv', args.data_dir / 'test.csv']}}, out / 'reproducibility.json')
    print(pd.DataFrame(rows).to_string(index=False))
    print(f'Selected using CV only: {selected}. Results: {out.resolve()}')


if __name__ == '__main__':
    main()
