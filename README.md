# Credit-Default-Risk-Prediction
Predicting loan default using Logistic Regression, Random Forest and XGBoost with SHAP analysis

## Project Overview
This project builds and compares three machine learning model to predict loan default risk, using a dataset of 255,347 loan applicants. The analysis includes exploratory data analysis(EDA), model comparison, and SHAP-based interpretability anlysis.

## key Findings
-XGBoost with class rebalancing achieved the best recall(62%) for identifying defaulters
-Age, Interest Rate, and Months Employed are the strongest prdictors of default
-Surprisingly, CreditScore and DTI Ratio showed limited predictive power in this dataset, suggesting traditional credit metrics alone maybe insufficient for default prediction.

## Models Compared
| Model | AUC | Recall (Default) |
|-------|-----|-----------------|
| Logistic Regression | 0.7484 | 0.03 |
| Random Forest | 0.7347 | 0.05 |
| XGBoost (rebalanced) | 0.7408 | 0.62 |

## Tech Stack
- Python (Pandas, NumPy, Matplotlib, Seaborn)
- Scikit-learn
- XGBoost
- SHAP

## Project Structure
-Predicting loan default.ipnb-Full analysis notebook

## Dataset
[Loan Default Dataset](https://www.kaggle.com/datasets/nikhil1e9/loan-default) from Kaggle
