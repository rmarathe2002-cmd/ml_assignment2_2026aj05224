# <Your Project Title>

## a. Problem statement
<2-4 lines: what you predict, why it matters, binary/multi-class.>

## b. Dataset description
- Source: <Kaggle/UCI link>
- Instances: <n> | Features: <n> | Target: `<column>` | Classes: <list>
- Preprocessing: <missing values, encoding, scaling>
- Split: 80/20 stratified, random_state=42

## c. GitHub repository
<your repo link>

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | | | | | | |
| Decision Tree | | | | | | |
| KNN | | | | | | |
| Naive Bayes | | | | | | |
| Support Vector Machine | | | | | | |

(Paste the table printed by `python train_models.py`.)

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | <your own analysis> |
| Decision Tree | |
| KNN | |
| Naive Bayes | |
| Support Vector Machine | |
| **Overall winner** | <model + why, referring to MCC/AUC/recall trade-offs> |

## Run locally
```
pip install -r requirements.txt
python train_models.py --data data.csv --target <target_col> --drop <id_col1,id_col2>
streamlit run app.py
```
`train_models.py` accepts any CSV (numeric + categorical columns, missing values) and
rejects datasets with fewer than 12 features or 500 instances. If `--target` is omitted,
the last column is used. Commit `model/` and `test_data.csv` after training.
