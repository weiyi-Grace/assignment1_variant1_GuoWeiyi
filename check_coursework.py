"""Training-only EDA supplement and independent checks of saved evaluation.

Run after train.py: python check_coursework.py --data-dir D:/dataset
Does not refit models or change model selection.
"""
import argparse
from pathlib import Path
import train as t


def explore(frame, out):
    numeric = frame.drop(columns='label').select_dtypes(include='number').columns.tolist()
    categorical = [c for c in frame if c not in numeric and c != 'label']
    rows = []
    for col in categorical:
        groups = frame.assign(category=frame[col].fillna('<MISSING>')).groupby('category')['label']
        for category, labels in groups:
            rows.append({'feature': col, 'category': category, 'count': len(labels),
                'fraction': len(labels) / len(frame), 'yes_rate': (labels == 'yes').mean()})
    t.pd.DataFrame(rows).to_csv(out / 'category_frequencies.csv', index=False)
    frame.groupby('label')[numeric].agg(['count', 'mean', 'median', 'std']).to_csv(out / 'numeric_by_class.csv')
    corr = frame[numeric].corr()  # Pairwise-complete training observations; descriptive only.
    corr.to_csv(out / 'numeric_correlations.csv')
    fig, ax = t.plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(numeric)), numeric, rotation=45, ha='right')
    ax.set_yticks(range(len(numeric)), numeric)
    for i in range(len(numeric)):
        for j in range(len(numeric)):
            ax.text(j, i, f'{corr.iloc[i,j]:.2f}', ha='center', va='center', fontsize=9)
    ax.set_title('Training numerical features: Pearson correlation')
    fig.colorbar(im, ax=ax, shrink=.75)
    fig.tight_layout(); fig.savefig(out / 'numeric_correlations.png', dpi=150); t.plt.close(fig)
    fig, axes = t.plt.subplots(2, 4, figsize=(13, 7))
    for col, ax in zip(numeric, axes.flat):
        ax.boxplot([frame.loc[frame.label == label, col].dropna() for label in ['no', 'yes']],
            tick_labels=['no', 'yes'], flierprops={'markersize': 1, 'alpha': .2})
        ax.set_title(col)
    axes.flat[-1].axis('off')
    fig.suptitle('Training distributions by label (outliers retained)')
    fig.tight_layout(); fig.savefig(out / 'boxplots_by_class.png', dpi=150); t.plt.close(fig)
    fig, ax = t.plt.subplots(figsize=(7, 5))
    for label, color in [('no', '#257c9e'), ('yes', '#da7633')]:
        sample = frame[frame.label == label].sample(frac=.1, random_state=t.SEED)
        ax.scatter(sample['stability_index'], sample['composite_rank'], s=6, alpha=.25, label=label, color=color)
    ax.set(xlabel='stability_index', ylabel='composite_rank', title='Training scatter: 10% sample within each class')
    ax.legend(); fig.tight_layout(); fig.savefig(out / 'training_scatter.png', dpi=150); t.plt.close(fig)


def verify(out, data_dir):
    predictions = t.pd.read_csv(out / 'test_predictions.csv')
    test = t.load_data(data_dir / 'test.csv').dropna(subset=['label'])
    assert t.np.array_equal(predictions.csv_row_number, test.index + 2), 'Row alignment mismatch'
    assert t.np.array_equal(predictions.actual, test.label), 'Actual label mismatch'
    y = (predictions.actual == 'yes').to_numpy()
    comparison = t.pd.read_csv(out / 'model_comparison.csv').set_index('model')
    checks = {}
    for name in comparison.index:
        pred = (predictions[name + '_prediction'] == 'yes').to_numpy()
        tp = int((y & pred).sum()); tn = int((~y & ~pred).sum())
        fp = int((~y & pred).sum()); fn = int((y & ~pred).sum())
        values = {'accuracy': (tp + tn) / len(y), 'precision_yes': tp / (tp + fp) if tp + fp else 0,
            'recall_yes': tp / (tp + fn), 'f1_yes': 2 * tp / (2 * tp + fp + fn),
            'balanced_accuracy': .5 * (tp / (tp + fn) + tn / (tn + fp))}
        for metric, value in values.items():
            assert t.np.isclose(value, comparison.loc[name, metric]), (name, metric)
        if name != 'majority_baseline':
            model = t.joblib.load(out / f'{name}.joblib')
            X = test.drop(columns='label')
            assert t.np.array_equal(model.predict(X), pred.astype(int))
            assert t.np.allclose(model.predict_proba(X)[:, list(model.classes_).index(1)],
                predictions[name + '_probability_yes'])
        checks[name] = {'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp, 'independent_metrics': values}
    # Check chosen rows against the stored full CV grid, without another search.
    tuning = t.json.loads((out / 'tuning.json').read_text(encoding='utf-8'))
    for name, result in tuning['models'].items():
        cv = t.pd.read_csv(out / f'{name}_cv_results.csv')
        assert t.np.isclose(cv.mean_test_f1.max(), result['cv_f1_mean'])
    t.save_json({'checks_passed': True, 'test_rows': len(y), 'checks': checks}, out / 'verification.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('D:/dataset'))
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'outputs')
    args = parser.parse_args()
    frame = t.load_data(args.data_dir / 'train.csv').dropna(subset=['label']).drop_duplicates()
    explore(frame, args.output_dir)
    verify(args.output_dir, args.data_dir)
    print('Training EDA exported. Saved predictions, metrics and CV scores verified.')


if __name__ == '__main__':
    main()
