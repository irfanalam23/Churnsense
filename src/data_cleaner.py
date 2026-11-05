import pandas as pd


def clean_dataset(df):

    # Create a copy of the original dataset
    cleaned_df = df.copy()

    # Clean column names
    cleaned_df.columns = [
        str(column).strip()
        for column in cleaned_df.columns
    ]

    # Count and remove duplicate rows
    duplicates_removed = int(cleaned_df.duplicated().sum())

    cleaned_df = cleaned_df.drop_duplicates()

    # Remove extra spaces from text values
    text_columns = cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        cleaned_df[column] = cleaned_df[column].apply(
            lambda value: value.strip()
            if isinstance(value, str)
            else value
        )

    # Count remaining missing values
    missing_values = int(cleaned_df.isnull().sum().sum())

    # Create cleaning report
    cleaning_report = {
        "Original Rows": len(df),
        "Cleaned Rows": len(cleaned_df),
        "Duplicate Rows Removed": duplicates_removed,
        "Missing Values Remaining": missing_values
    }

    return cleaned_df, cleaning_report