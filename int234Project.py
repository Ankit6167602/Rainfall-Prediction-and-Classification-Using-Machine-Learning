
# ============================================================
# RAINFALL PREDICTION USING SIX MACHINE LEARNING MODELS
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
    OneHotEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

import os
import warnings

warnings.filterwarnings("ignore")

os.makedirs("results", exist_ok=True)


# 2. LOAD DATASET

df = pd.read_csv(
    r"C:\Users\ankit\Downloads\Sub_Division_IMD_2017.csv"
)

print("\nOriginal Dataset")
print(df.head())

print("\nOriginal Shape:", df.shape)


# 3. SELECT REQUIRED COLUMNS

columns = [
    "SUBDIVISION",
    "YEAR",
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "ANNUAL"
]

df = df[columns].copy()


# 4. DATA PREPROCESSING

numeric_columns = [
    "YEAR", "JAN", "FEB", "MAR",
    "APR", "MAY", "ANNUAL"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=columns)

df = df.drop_duplicates()

df["EARLY_RAINFALL"] = df[
    ["JAN", "FEB", "MAR", "APR", "MAY"]
].sum(axis=1)

df = df.sort_values("YEAR").reset_index(drop=True)

print("\nCleaned Dataset Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

df.to_csv("results/cleaned_rainfall.csv", index=False)


# 5. CHRONOLOGICAL TRAIN-TEST SPLIT

unique_years = np.sort(df["YEAR"].unique())

split_position = int(len(unique_years) * 0.80)

split_year = unique_years[split_position]

train_df = df[df["YEAR"] < split_year].copy()
test_df = df[df["YEAR"] >= split_year].copy()

print("\nTraining data:", train_df.shape)
print("Testing data:", test_df.shape)

print("Training until year:", split_year - 1)
print("Testing from year:", split_year)

y_train = train_df["ANNUAL"]
y_test = test_df["ANNUAL"]


# ============================================================
# PART A: REGRESSION MODELS
# ============================================================

regression_results = []


# COMMON REGRESSION EVALUATION FUNCTION

def evaluate_regression(name, model, X_train, X_test):

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    regression_results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    print("\n", name)
    print("MAE:", round(mae, 2))
    print("MSE:", round(mse, 2))
    print("RMSE:", round(rmse, 2))
    print("R2 Score:", round(r2, 4))

    return predictions


# 6. MODEL 1: SIMPLE LINEAR REGRESSION

X_train_simple = train_df[["EARLY_RAINFALL"]]
X_test_simple = test_df[["EARLY_RAINFALL"]]

simple_model = LinearRegression()

pred_simple = evaluate_regression(
    "Simple Linear Regression",
    simple_model,
    X_train_simple,
    X_test_simple
)


# 7. MODEL 2: MULTIPLE LINEAR REGRESSION

numeric_features = [
    "YEAR", "JAN", "FEB", "MAR", "APR", "MAY"
]

categorical_features = ["SUBDIVISION"]

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", numeric_features),
        ("categorical",
         OneHotEncoder(handle_unknown="ignore"),
         categorical_features)
    ]
)

X_train_multiple = train_df[
    numeric_features + categorical_features
]

X_test_multiple = test_df[
    numeric_features + categorical_features
]

multiple_model = Pipeline([
    ("preprocessing", preprocessor),
    ("regression", LinearRegression())
])

pred_multiple = evaluate_regression(
    "Multiple Linear Regression",
    multiple_model,
    X_train_multiple,
    X_test_multiple
)


# 8. MODEL 3: POLYNOMIAL REGRESSION

polynomial_model = Pipeline([
    ("polynomial", PolynomialFeatures(
        degree=2,
        include_bias=False
    )),
    ("regression", LinearRegression())
])

pred_polynomial = evaluate_regression(
    "Polynomial Regression",
    polynomial_model,
    X_train_simple,
    X_test_simple
)


# 9. REGRESSION PERFORMANCE TABLE

regression_df = pd.DataFrame(regression_results)

regression_df = regression_df.sort_values("RMSE")

print("\nREGRESSION COMPARISON")
print(regression_df.to_string(index=False))

regression_df.to_csv(
    "results/regression_comparison.csv",
    index=False
)


# 10. REGRESSION COMPARISON GRAPH

plt.figure(figsize=(9, 5))

sns.barplot(
    data=regression_df,
    x="Model",
    y="RMSE"
)

plt.title("Regression Models - RMSE Comparison")
plt.xticks(rotation=20)
plt.ylabel("RMSE (mm)")
plt.tight_layout()

plt.savefig("results/regression_comparison.png")
plt.show()


# 11. ACTUAL VS PREDICTED GRAPH

plt.figure(figsize=(8, 6))

plt.scatter(y_test, pred_simple, label="Simple Linear")
plt.scatter(y_test, pred_multiple, label="Multiple Linear")
plt.scatter(y_test, pred_polynomial, label="Polynomial")

min_value = min(
    y_test.min(),
    pred_simple.min(),
    pred_multiple.min(),
    pred_polynomial.min()
)

max_value = max(
    y_test.max(),
    pred_simple.max(),
    pred_multiple.max(),
    pred_polynomial.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="black",
    label="Perfect Prediction"
)

plt.xlabel("Actual Annual Rainfall (mm)")
plt.ylabel("Predicted Annual Rainfall (mm)")
plt.title("Actual vs Predicted Rainfall")
plt.legend()
plt.tight_layout()

plt.savefig("results/actual_vs_predicted.png")
plt.show()


# ============================================================
# PART B: CLASSIFICATION MODELS
# ============================================================

# 12. CREATE RAINFALL CATEGORIES

low_threshold = y_train.quantile(0.33)
high_threshold = y_train.quantile(0.67)

def categorize_rainfall(values):

    return pd.cut(
        values,
        bins=[
            -np.inf,
            low_threshold,
            high_threshold,
            np.inf
        ],
        labels=["Low", "Moderate", "High"],
        include_lowest=True
    )


y_train_class = categorize_rainfall(y_train)
y_test_class = categorize_rainfall(y_test)

print("\nRainfall Categories:")
print("Low: Up to", round(low_threshold, 2), "mm")
print(
    "Moderate:",
    round(low_threshold, 2),
    "to",
    round(high_threshold, 2),
    "mm"
)
print("High: Above", round(high_threshold, 2), "mm")


# 13. CLASSIFICATION INPUTS

X_train_class = train_df[
    numeric_features + categorical_features
]

X_test_class = test_df[
    numeric_features + categorical_features
]


# COMMON CLASSIFICATION PREPROCESSOR

classification_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features
        ),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# 14. CLASSIFICATION EVALUATION

classification_results = []

classification_predictions = {}


def evaluate_classification(name, model):

    model.fit(X_train_class, y_train_class)

    predictions = model.predict(X_test_class)

    classification_predictions[name] = predictions

    accuracy = accuracy_score(y_test_class, predictions)

    precision = precision_score(
        y_test_class,
        predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_test_class,
        predictions,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_test_class,
        predictions,
        average="macro",
        zero_division=0
    )

    classification_results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("\n", name)

    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))

    print("\nClassification Report:")

    print(
        classification_report(
            y_test_class,
            predictions,
            labels=["Low", "Moderate", "High"],
            zero_division=0
        )
    )


# 15. MODEL 4: KNN

knn_model = Pipeline([
    ("preprocessing", classification_preprocessor),
    ("classifier", KNeighborsClassifier(n_neighbors=5))
])

evaluate_classification(
    "K-Nearest Neighbors",
    knn_model
)


# 16. MODEL 5: DECISION TREE

decision_tree_model = Pipeline([
    ("preprocessing", classification_preprocessor),
    ("classifier", DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ))
])

evaluate_classification(
    "Decision Tree",
    decision_tree_model
)


# 17. MODEL 6: LOGISTIC REGRESSION

logistic_model = Pipeline([
    ("preprocessing", classification_preprocessor),
    ("classifier", LogisticRegression(
        max_iter=2000,
        random_state=42
    ))
])

evaluate_classification(
    "Logistic Regression",
    logistic_model
)


# 18. CLASSIFICATION PERFORMANCE TABLE

classification_df = pd.DataFrame(classification_results)

classification_df = classification_df.sort_values(
    "F1 Score",
    ascending=False
)

print("\nCLASSIFICATION COMPARISON")
print(classification_df.to_string(index=False))

classification_df.to_csv(
    "results/classification_comparison.csv",
    index=False
)


# 19. CLASSIFICATION COMPARISON GRAPH

plt.figure(figsize=(9, 5))

sns.barplot(
    data=classification_df,
    x="Model",
    y="Accuracy"
)

plt.title("Classification Models - Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()

plt.savefig("results/classification_comparison.png")
plt.show()


# 20. CONFUSION MATRICES

for name, predictions in classification_predictions.items():

    cm = confusion_matrix(
        y_test_class,
        predictions,
        labels=["Low", "Moderate", "High"]
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low", "Moderate", "High"],
        yticklabels=["Low", "Moderate", "High"]
    )

    plt.title(name + " - Confusion Matrix")
    plt.xlabel("Predicted Category")
    plt.ylabel("Actual Category")
    plt.tight_layout()

    filename = name.lower().replace(" ", "_")

    plt.savefig(
        "results/" + filename + "_confusion_matrix.png"
    )

    plt.show()


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n===================================")
print("ALL SIX MODELS TRAINED SUCCESSFULLY")
print("===================================")

print("\nRegression Results:")
print(regression_df)

print("\nClassification Results:")
print(classification_df)

print("\nAll result files are saved in the results folder.")


