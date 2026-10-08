# M9 vs External Dataset Comparison — M12.1

> **Important:** M9 was evaluated on the original locked test split of the project dataset.
> The External dataset is a separately sourced image collection covering overlapping Tomato disease classes.
> These are **not the same data split**. This comparison is informational only.

## Metric Comparison Table

| Metric | M9 Locked Test | External Tomato | Difference |
|--------|---------------|-----------------|------------|
| Accuracy | 95.558% | 95.691% | +0.13 pp |
| Macro Precision | 94.003% | 95.021% | +1.02 pp |
| Macro Recall | 94.427% | 95.274% | +0.85 pp |
| Macro F1 | 94.063% | 95.034% | +0.97 pp |
| Weighted F1 | 95.548% | 95.666% | +0.12 pp |

*pp = percentage-point difference*

## Interpretation
- A positive difference indicates the external dataset yielded **higher** performance than M9.
- A negative difference indicates **lower** external performance.
- Differences may reflect distribution shift between the datasets, not a change in the model.
- The model was **not retrained** or modified based on these results.
- M9 remains the **authoritative locked-test evaluation**.
