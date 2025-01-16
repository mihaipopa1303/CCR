import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import (
    LassoCV,
    RidgeClassifierCV,
    LogisticRegression,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    precision_score,
    recall_score,
    make_scorer,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer


#  VIF function:
def calculate_vif(dataframe):
    vif_data = pd.DataFrame()
    vif_data["Feature"] = dataframe.columns
    vif_data["VIF"] = [
        variance_inflation_factor(dataframe.values, i)
        for i in range(dataframe.shape[1])
    ]
    return vif_data

file_path = '/Users/mihaipopa/Library/CloudStorage/OneDrive-CorPrime/Documents/Crypto_Counterparty_Dataset.csv'
dataset = pd.read_csv(file_path)

counterparty_names = dataset["Counterparty"].copy()
X = dataset.drop(columns=["Default", "Counterparty"])
y = dataset["Default"]  # Target Variable set as late payments 30 days,

print("Full dataset shape (features):", X.shape)
print("Full target distribution:\n", y.value_counts(normalize=True))


# Train-Test Split:

X_train, X_test, y_train, y_test, cpart_train, cpart_test = train_test_split(X,y,
    counterparty_names,
    test_size=0.3,
    random_state=42,
    stratify=y  # imbalanced data
)

print("\nTrain shape:", X_train.shape, " Test shape:", X_test.shape)
print("Train target distribution:\n", y_train.value_counts(normalize=True))
print("Test target distribution:\n", y_test.value_counts(normalize=True))

numeric_cols = X_train.select_dtypes(include=[np.number]).columns
X_train_numeric = X_train[numeric_cols].copy()
X_test_numeric = X_test[numeric_cols].copy()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_numeric)
X_test_scaled = scaler.transform(X_test_numeric)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=numeric_cols, index=X_train.index)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=numeric_cols, index=X_test.index)


# Drop features with high VIF (above 5)

vif_data = calculate_vif(X_train_scaled_df)
low_vif_features = vif_data[vif_data["VIF"] <= 5]["Feature"].tolist()

print("\nVIF Data:\n", vif_data)
print("\nFeatures that passed the VIF test (VIF <= 5):")
print(low_vif_features)

X_train_reduced = X_train_scaled_df[low_vif_features]
X_test_reduced = X_test_scaled_df[low_vif_features]


# Ensemble feature selection:

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "Bagging": BaggingClassifier(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    "Lasso": LassoCV(cv=5, random_state=42),
    "Ridge": RidgeClassifierCV(cv=5),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42,use_label_encoder=False, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(n_estimators=100, random_state=42),
    "CatBoost": CatBoostClassifier(n_estimators=100, random_state=42, verbose=0),
}

feature_importances = pd.DataFrame(index=low_vif_features)

for model_name, model in models.items():
    try:
        model.fit(X_train_reduced, y_train)
        if hasattr(model, "feature_importances_"):
            feature_importances[model_name] = model.feature_importances_
        elif hasattr(model, "coef_"):
            feature_importances[model_name] = np.abs(model.coef_).ravel()
        else:
            print(f"{model_name} does not support feature importance extraction.")
    except Exception as e:
        print(f"Error training {model_name}: {e}")

feature_importances["Mean_Importance"] = feature_importances.mean(axis=1)
feature_importances.sort_values("Mean_Importance", ascending=False, inplace=True)

importance_file = '/Users/mihaipopa/Library/CloudStorage/OneDrive-CorPrime/Documents/Feature_Importances_By_Model.csv'
feature_importances.to_csv(importance_file)
print(f"\nFeature importances by model saved to: {importance_file}")

mean_val = feature_importances["Mean_Importance"].mean()
top_features = feature_importances[feature_importances["Mean_Importance"] > mean_val].index.tolist()

print("\nTop Features Selected (Mean Importance > Mean):")
print(top_features)

X_train_top = X_train_reduced[top_features]
X_test_top = X_test_reduced[top_features]


# For model evaluation and results:

def evaluate_model(y_true, y_pred, y_pred_prob, title=""):
    print(f"\n{title}")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("\nClassification Report:\n", classification_report(y_true, y_pred))
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"ROC AUC Score: {roc_auc_score(y_true, y_pred_prob):.4f}")


# Logistic Regression with Bayesian Optimization

print("Before Logistic Regression Training:")
print("X_train_top shape:", X_train_top.shape)
print("X_test_top shape :", X_test_top.shape)

param_space_lr = [
    {
        "C": Real(0.001, 10, prior="log-uniform"),
        "penalty": Categorical(["l1", "l2"]),
        "solver": Categorical(["liblinear", "saga"]),
        "max_iter": Integer(500, 5000),
    },
    {
        "C": Real(0.001, 10, prior="log-uniform"),
        "penalty": Categorical(["elasticnet"]),
        "solver": Categorical(["saga"]),
        "l1_ratio": Real(0, 1, prior="uniform"),
        "max_iter": Integer(500, 5000),
    },
]

bayes_search_lr = BayesSearchCV(
    LogisticRegression(random_state=42),
    search_spaces=param_space_lr,
    scoring="recall",
    n_iter=50,
    cv=5,
    random_state=42,
    verbose=1,
    n_jobs=-1,
)

bayes_search_lr.fit(X_train_top, y_train)
print("\nBest parameters for Logistic Regression with Bayesian Optimization:")
print(bayes_search_lr.best_params_)

best_lr = bayes_search_lr.best_estimator_
best_lr.fit(X_train_top, y_train)

y_pred_lr = best_lr.predict(X_test_top)
y_pred_prob_lr = best_lr.predict_proba(X_test_top)[:, 1]

evaluate_model(y_test, y_pred_lr, y_pred_prob_lr,
               title="Bayesian Optimized Logistic Regression")

lr_output = pd.DataFrame({
    "Counterparty": cpart_test,  # from test-split
    "Default_Probability": y_pred_prob_lr,
    "Predicted_Class": y_pred_lr,
    "Actual_Default": y_test
})
lr_output_file = '/Users/mihaipopa/Library/CloudStorage/OneDrive-CorPrime/Documents/Logistic_Regression_Predicted_Probabilities.csv'
lr_output.to_csv(lr_output_file, index=False)
print(f"\nLogistic Regression predictions saved to: {lr_output_file}")


# Random Forest with Bayesian Optimization (there is a problem here, there might be data leakage and need to check!!)

# print("\nBefore Random Forest Training:")
# print("X_train_top shape:", X_train_top.shape)
# print("X_test_top shape :", X_test_top.shape)
#
# param_space_rf = {
#     "n_estimators": Integer(50, 300),
#     "max_depth": Integer(5, 50),
#     "min_samples_split": Integer(2, 20),
#     "min_samples_leaf": Integer(1, 10),
# }
#
# bayes_search_rf = BayesSearchCV(
#     RandomForestClassifier(random_state=42),
#     search_spaces=param_space_rf,
#     scoring=make_scorer(recall_score),
#     n_iter=50,
#     cv=5,
#     random_state=42,
#     verbose=1,
#     n_jobs=-1,
# )
#
# bayes_search_rf.fit(X_train_top, y_train)
# print("\nBest parameters for Random Forest with Bayesian Optimization:")
# print(bayes_search_rf.best_params_)
#
# best_rf = bayes_search_rf.best_estimator_
# best_rf.fit(X_train_top, y_train)
#
# y_pred_rf = best_rf.predict(X_test_top)
# y_pred_prob_rf = best_rf.predict_proba(X_test_top)[:, 1]
#
# evaluate_model(y_test, y_pred_rf, y_pred_prob_rf,
#                title="Bayesian Optimized Random Forest")
#
# rf_output = pd.DataFrame({
#     "Counterparty": cpart_test,  # from test-split
#     "Default_Probability": y_pred_prob_rf,
#     "Predicted_Class": y_pred_rf,
#     "Actual_Default": y_test
# })
# rf_output_file = '/Users/mihaipopa/Library/CloudStorage/OneDrive-CorPrime/Documents/Random_Forest_Predicted_Probabilities.csv'
# rf_output.to_csv(rf_output_file, index=False)
# print(f"\nRandom Forest predictions saved to: {rf_output_file}")


