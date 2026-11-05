import pandas as pd


def analyze_dataset(df):
    """
    Analyze a customer dataset and generate useful statistics.
    """

    # Store analysis results
    analysis = {}

    # 1. Basic Dataset Information
    analysis["total_customers"] = len(df)
    analysis["total_columns"] = len(df.columns)

    # 2. Missing Values
    analysis["missing_values"] = int(df.isnull().sum().sum())

    # 3. Duplicate Rows
    analysis["duplicate_rows"] = int(df.duplicated().sum())

    # 4. Numerical Analysis
    numerical_columns = df.select_dtypes(
        include="number"
    ).columns

    numerical_summary = df[numerical_columns].describe()

    # 5. Churn Analysis
    churn_column = None

    for column in df.columns:
        if column.strip().lower() in [
            "churn",
            "exited",
            "attrition",
            "customer_churn"
        ]:
            churn_column = column
            break

    if churn_column is not None:

        churn_counts = df[churn_column].value_counts()

        analysis["churn_column"] = churn_column
        analysis["churn_counts"] = churn_counts.to_dict()

        # Identify common representations of churn
        churn_values = df[churn_column].astype(str).str.strip().str.lower()

        churned_count = int(
            churn_values.isin(
                ["yes", "true", "1", "churned", "exited"]
            ).sum()
        )

        analysis["churned_customers"] = churned_count

        if len(df) > 0:
            analysis["churn_rate"] = round(
                (churned_count / len(df)) * 100,
                2
            )
        else:
            analysis["churn_rate"] = 0

    else:

        analysis["churn_column"] = None
        analysis["churn_counts"] = {}
        analysis["churned_customers"] = None
        analysis["churn_rate"] = None

    # 6. Revenue Analysis
    revenue_column = None

    for column in df.columns:
        if column.strip().lower() in [
            "revenue",
            "totalcharges",
            "total_charges",
            "sales",
            "monthlycharges",
            "monthly_charges"
        ]:
            revenue_column = column
            break

    if revenue_column is not None:

        revenue_values = pd.to_numeric(
            df[revenue_column],
            errors="coerce"
        )

        analysis["revenue_column"] = revenue_column
        analysis["total_revenue"] = round(
            float(revenue_values.sum()),
            2
        )
        analysis["average_revenue"] = round(
            float(revenue_values.mean()),
            2
        )

    else:

        analysis["revenue_column"] = None
        analysis["total_revenue"] = None
        analysis["average_revenue"] = None

    # Return analysis results
    return analysis, numerical_summary