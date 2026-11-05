import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def train_churn_model(df):
    """
    Train a machine learning model to predict customer churn.
    """

    data = df.copy()

    # Find the churn target column
    churn_column = None

    for column in data.columns:
        if column.strip().lower() in [
            "churn",
            "exited",
            "attrition",
            "customer_churn"
        ]:
            churn_column = column
            break

    if churn_column is None:
        raise ValueError(
            "Churn column not found in the dataset."
        )

    # Remove rows where the target value is missing
    data = data.dropna(subset=[churn_column])

    # Separate features and target
    X = data.drop(columns=[churn_column])
    y = data[churn_column].astype(str).str.strip().str.lower()

    # Convert common churn labels to binary values
    positive_labels = {
        "yes", "true", "1", "churned", "exited"
    }

    negative_labels = {
        "no", "false", "0", "not churned", "stayed"
    }

    y = y.map(
        lambda value: (
            1 if value in positive_labels
            else 0 if value in negative_labels
            else None
        )
    )

    # Remove rows with unsupported target labels
    valid_rows = y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows].astype(int)

    if y.nunique() != 2:
        raise ValueError(
            "The churn column must contain both churn and non-churn examples."
        )

    if len(X) < 10:
        raise ValueError(
            "The dataset is too small to train a useful model. "
            "Please upload more customer records."
        )

    # Remove columns that are completely empty
    X = X.dropna(axis=1, how="all")

    # Remove identifier columns that usually don't help prediction
    identifier_columns = [
        column for column in X.columns
        if column.strip().lower() in [
            "customerid",
            "customer_id",
            "customer id",
            "id"
        ]
    ]

    X = X.drop(columns=identifier_columns, errors="ignore")

    if X.shape[1] == 0:
        raise ValueError(
            "No usable feature columns remain after removing identifiers."
        )

    # Identify numerical and categorical columns
    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Preprocess numerical columns
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # Preprocess categorical columns
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, numerical_columns),
            ("categorical", categorical_pipeline, categorical_columns)
        ]
    )

    # Create machine learning pipeline
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced"
                )
            )
        ]
    )

    # Split data into training and testing sets
    class_counts = y.value_counts()

    stratify_target = (
        y if class_counts.min() >= 2 else None
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_target
    )

    # Train the model
    model.fit(X_train, y_train)

    # Generate predictions
    predictions = model.predict(X_test)

    # Evaluate the model
    metrics = {
        "accuracy": round(
            accuracy_score(y_test, predictions) * 100, 2
        ),
        "precision": round(
            precision_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "recall": round(
            recall_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "f1_score": round(
            f1_score(
                y_test, predictions, zero_division=0
            ) * 100, 2
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=[0, 1]
        ).tolist(),
        "training_rows": len(X_train),
        "testing_rows": len(X_test)
    }

    return model, metrics