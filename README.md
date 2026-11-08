## a. Problem statement
Develop a machine learning classification model to predict whether a bank customer will subscribe to a term deposit based on their demographic information, financial status, and previous marketing campaign interactions.

## b. Dataset description
The dataset contains customer and campaign-related attributes such as:
Age
Job type
Marital status
Education level
Credit default status
Account balance
Housing loan status
Personal loan status
Contact communication type
Last contact day and month
Call duration
Number of campaign contacts
Days since previous contact (pdays)
Number of previous contacts (previous)
Outcome of previous marketing campaign (poutcome)
Target Variable
deposit (Yes/No)
Yes → Customer subscribed to the term deposit.
No → Customer did not subscribe to the term deposit.

## c. GitHub repository
https://github.com/rmarathe2002-cmd/ml_assignment2_2026aj05224.git

## d. Models used

| ML Model Name          |   Accuracy |    AUC |   Precision |   Recall |     F1 |    MCC |
|:-----------------------|-----------:|-------:|------------:|---------:|-------:|-------:|
| Logistic Regression    |     0.8262 | 0.9071 |      0.8278 |   0.7996 | 0.8135 | 0.6513 |
| Decision Tree          |     0.811  | 0.8798 |      0.8093 |   0.7864 | 0.7977 | 0.6207 |
| KNN                    |     0.8222 | 0.8881 |      0.8282 |   0.7883 | 0.8077 | 0.6433 |
| Naive Bayes            |     0.7358 | 0.8142 |      0.7799 |   0.6163 | 0.6885 | 0.4742 |
| Support Vector Machine |     0.8554 | 0.9194 |      0.8249 |   0.8819 | 0.8524 | 0.7124 |

### Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Second-best overall, with accuracy 0.8262, AUC 0.9071 and MCC 0.6513. Its precision (0.8278) and recall (0.7996) are close together, so it is balanced, though it misses somewhat more positives than it falsely flags. Its strong AUC suggests the classes are largely separable by a linear boundary. It is fast and interpretable, but the SVM beats it on recall and MCC.|
| Decision Tree | Accuracy 0.811, AUC 0.8798, MCC 0.6207. It is the weakest of the three mid-ranked models on AUC, which likely reflects the unstable, step-like probability estimates of a single tree. Precision (0.8093) and recall (0.7864) are balanced, and it needs no scaling, but it trails the margin-based models in generalisation.|
| KNN | Accuracy 0.8222 and MCC 0.6433, close to Logistic Regression. It has the highest precision of all models (0.8282), but recall is only 0.7883, so it is slightly conservative about predicting the positive class. Its AUC (0.8881) is lower than LR and SVM, likely because distance-based voting gives coarser probability scores.|
| Naive Bayes | The weakest model on every metric: accuracy 0.7358, AUC 0.8142, MCC 0.4742. Its recall of 0.6163 is far below its precision of 0.7799, so it misses many positive cases. This is probably due to its assumption that features are independent, which is unlikely to hold fully for this data, especially with one-hot encoded or correlated features. It is very fast, but not suitable here if missing positives is costly.|
| Support Vector Machine | The best model on most metrics: accuracy 0.8554, AUC 0.9194, recall 0.8819, F1 0.8524 and MCC 0.7124. Its recall is notably higher than its precision (0.8249), so it catches more positives at the cost of a few more false alarms. The RBF kernel likely captures non-linear patterns that Logistic Regression misses. The cost is slower training and lower interpretability.|
| **Overall winner** | Support Vector Machine. It has the highest accuracy, AUC, recall, F1 and MCC. Its MCC of 0.7124 is about 0.06 above the runner-up (Logistic Regression, 0.6513), and MCC is a reliable single score because it uses all four confusion-matrix cells. KNN has slightly higher precision (0.8282 vs 0.8249), but the SVM’s much better recall (0.8819 vs 0.7883) makes it the better all-round choice. Logistic Regression is a good simpler alternative if interpretability or speed matters.|
