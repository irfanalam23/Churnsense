
import streamlit as st
import pandas as pd
from src.data_cleaner import clean_dataset
from src.data_analyzer import analyze_dataset
from src.churn_predictor import train_churn_model
from src.llm_service import generate_ai_insights
# ----------------------------------
# 1. PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="ChurnSense",
    page_icon="📊",
    layout="wide"
)

# ----------------------------------
# 2. APPLICATION TITLE
# ----------------------------------

st.title("📊 ChurnSense")

st.subheader("AI-Powered Customer Churn Intelligence")

st.write(
    "Upload your customer dataset to analyze customer behavior, "
    "identify churn patterns, and generate useful business insights."
)

# ----------------------------------
# 3. SIDEBAR NAVIGATION
# ----------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Choose a page",
    [
        "Home",
        "Dataset Analysis",
         "Interactive Dashboard",
        "Churn Prediction",
        "AI Insights"
    ]
)

# ----------------------------------
# 4. HOME PAGE
# ----------------------------------

if page == "Home":

    st.header("Welcome to ChurnSense")

    st.write(
        """
        ChurnSense is a customer analytics platform that helps
        businesses understand customer behavior and churn.

        Upload a dataset to begin your analysis.
        """
    )

    st.success("ChurnSense is running successfully!")

    st.info(
        "Select Dataset Analysis from the sidebar "
        "to upload your customer dataset."
    )

# ----------------------------------
# 5. DATASET ANALYSIS PAGE
# ----------------------------------

elif page == "Dataset Analysis":

    st.header("📂 Dataset Upload")

    st.write(
        "Upload your customer dataset in CSV or Excel format."
    )

    uploaded_file = st.file_uploader(
        "Choose your dataset",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:

            # Read CSV or Excel file
            if uploaded_file.name.lower().endswith(".csv"):

                df = pd.read_csv(uploaded_file)

            else:

                df = pd.read_excel(uploaded_file)
            st.session_state["uploaded_df"] = df

            # Check whether the dataset has rows and columns
            if df.empty or len(df.columns) == 0:

                st.warning(
                    "The uploaded dataset is empty. "
                    "Please upload a dataset containing data."
                )

            else:

                st.success("Dataset uploaded successfully!")

                # Dataset preview
                st.subheader("📋 Dataset Preview")

                st.dataframe(
                    df.head(10),
                    use_container_width=True
                )

                # Dataset overview
                st.subheader("📊 Dataset Overview")

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Total Rows",
                        df.shape[0]
                    )

                with col2:

                    st.metric(
                        "Total Columns",
                        df.shape[1]
                    )

                with col3:

                    st.metric(
                        "Missing Values",
                        int(df.isnull().sum().sum())
                    )

                # Column information
                st.subheader("Column Information")

                column_info = pd.DataFrame({
                    "Column Name": df.columns.astype(str),
                    "Data Type": df.dtypes.astype(str).values,
                    "Missing Values": df.isnull().sum().values
                })

                st.dataframe(
                    column_info,
                    use_container_width=True
                )

                # Statistical summary
                st.subheader("📈 Numerical Summary")

                numerical_columns = df.select_dtypes(
                    include="number"
                ).columns

                if len(numerical_columns) > 0:

                    st.dataframe(
                        df[numerical_columns].describe(),
                        use_container_width=True
                    )

                else:

                    st.info(
                        "No numerical columns were found "
                        "in this dataset."
                    )

                                # ----------------------------------
                # DATA CLEANING
                # ----------------------------------

                st.subheader("🧹 Data Cleaning")

                st.write(
                    "Remove duplicate rows and extra spaces "
                    "from column names and text values."
                )

                duplicate_count = int(df.duplicated().sum())

                st.metric(
                    "Duplicate Rows Found",
                    duplicate_count
                )

                if st.button("Clean Dataset"):

                    cleaned_df, cleaning_report = clean_dataset(df)

                    st.session_state["cleaned_df"] = cleaned_df

                    st.success("Dataset cleaning completed!")

                    st.subheader("📊 Cleaning Report")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Original Rows",
                            cleaning_report["Original Rows"]
                        )

                    with col2:
                        st.metric(
                            "Cleaned Rows",
                            cleaning_report["Cleaned Rows"]
                        )

                    with col3:
                        st.metric(
                            "Duplicates Removed",
                            cleaning_report["Duplicate Rows Removed"]
                        )

                    st.write(
                        "Missing values remaining:",
                        cleaning_report["Missing Values Remaining"]
                    )

                    st.subheader("Cleaned Dataset Preview")

                    st.dataframe(
                        cleaned_df.head(10),
                        use_container_width=True
                    )

                    csv_data = cleaned_df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="Download Cleaned Dataset",
                        data=csv_data,
                        file_name="cleaned_dataset.csv",
                        mime="text/csv"
                    )

                # Display cleaned dataset after Streamlit reruns
                if "cleaned_df" in st.session_state:

                    st.subheader("Previously Cleaned Dataset")

                    st.dataframe(
                        st.session_state["cleaned_df"].head(10),
                        use_container_width=True
                    )

        except Exception as e:

            st.error(
                f"Unable to read the uploaded file: {e}"
            )
# ----------------------------------
# DATA ANALYTICS
# ----------------------------------

st.divider()

st.subheader("📊 Advanced Data Analytics")

if st.button("Analyze Dataset"):

    analysis, numerical_summary = analyze_dataset(df)

    st.session_state["analysis_results"] = analysis
    st.session_state["numerical_summary"] = numerical_summary

if "analysis_results" in st.session_state:

    analysis = st.session_state["analysis_results"]

    # Customer Overview
    st.subheader("👥 Customer Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Customers / Rows",
            analysis["total_customers"]
        )

    with col2:
        st.metric(
            "Total Columns",
            analysis["total_columns"]
        )

    with col3:
        st.metric(
            "Missing Values",
            analysis["missing_values"]
        )

    # Churn Analysis
    st.subheader("📉 Churn Analysis")

    if analysis["churn_column"] is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Churned Customers",
                analysis["churned_customers"]
            )

        with col2:
            st.metric(
                "Churn Rate",
                f'{analysis["churn_rate"]}%'
            )

        st.write("Customer Churn Distribution")

        churn_df = pd.DataFrame(
            list(analysis["churn_counts"].items()),
            columns=["Churn Status", "Customer Count"]
        )

        st.bar_chart(
            churn_df.set_index("Churn Status")
        )

    else:

        st.info(
            "No recognized churn column was found. "
            "Try a dataset containing a Churn or Exited column."
        )

    # Revenue Analysis
    st.subheader("💰 Revenue Analysis")

    if analysis["revenue_column"] is not None:

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Total Revenue / Charges",
                f'{analysis["total_revenue"]:,.2f}'
            )

        with col2:
            st.metric(
                "Average Revenue / Charges",
                f'{analysis["average_revenue"]:,.2f}'
            )

    else:

        st.info(
            "No recognized revenue or charges column was found."
        )

    # Numerical Summary
    st.subheader("📈 Numerical Summary")

    numerical_summary = st.session_state["numerical_summary"]

    if not numerical_summary.empty:

        st.dataframe(
            numerical_summary,
            use_container_width=True
        )

    else:

        st.info("No numerical columns found in the dataset.")

    # Data Quality
    st.subheader("🔍 Data Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Duplicate Rows",
            analysis["duplicate_rows"]
        )

    with col2:
        st.metric(
            "Missing Values",
            analysis["missing_values"]
        )
       
elif page == "Interactive Dashboard":

    st.header("📊 Interactive Customer Churn Dashboard")

    # Check whether a dataset has been uploaded
    if "uploaded_df" not in st.session_state:

        st.info(
            "Please upload a dataset from the Dataset Analysis page first."
        )

    else:

        # Use cleaned dataset if available, otherwise use uploaded data
        if "cleaned_df" in st.session_state:
            df = st.session_state["cleaned_df"]
        else:
            df = st.session_state["uploaded_df"]

        # Customer overview
        st.subheader("👥 Customer Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Customers", f"{len(df):,}")

        with col2:
            st.metric("Total Columns", len(df.columns))

        with col3:
            st.metric(
                "Missing Values",
                int(df.isnull().sum().sum())
            )

        st.divider()

        # Find the churn column
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

        if churn_column:

            churn_values = (
                df[churn_column]
                .astype(str)
                .str.strip()
                .str.lower()
            )

            churned_count = int(
                churn_values.isin(
                    ["yes", "true", "1", "churned", "exited"]
                ).sum()
            )

            churn_rate = (
                (churned_count / len(df)) * 100
                if len(df) > 0 else 0
            )

            # Churn KPIs
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Churned Customers", f"{churned_count:,}")

            with col2:
                st.metric("Customer Churn Rate", f"{churn_rate:.2f}%")

            st.divider()

            # Churn distribution chart
            st.subheader("📈 Churn Distribution")

            churn_counts = df[churn_column].astype(str).value_counts()

            st.bar_chart(churn_counts)

        else:
            st.warning(
                "Churn column not found in this dataset."
            )

        st.divider()

        # Numerical data analysis
        numerical_columns = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numerical_columns:

            st.subheader("📊 Numerical Data Analysis")

            selected_column = st.selectbox(
                "Choose a numerical column",
                numerical_columns
            )

            st.bar_chart(df[selected_column].dropna())

            st.subheader("📋 Statistical Summary")

            st.dataframe(
                df[numerical_columns].describe(),
                use_container_width=True
            )

        else:
            st.info("No numerical columns are available for analysis.")


# ----------------------------------
# 6. CHURN PREDICTION PAGE
# ----------------------------------
 #
elif page == "Churn Prediction":

    st.header("🔮 Customer Churn Prediction")

    st.write(
        "Train a machine learning model using your uploaded "
        "customer dataset."
    )

    if "uploaded_df" not in st.session_state:
        st.info(
            "Please upload a dataset from the Dataset Analysis page first."
        )

    else:

        if st.button("Train Churn Prediction Model"):

            try:
                df = st.session_state["uploaded_df"]

                with st.spinner("Training machine learning model..."):

                    model, metrics = train_churn_model(df)

                st.session_state["churn_model"] = model
                st.session_state["churn_metrics"] = metrics

                st.success("Model training completed!")

            except Exception as e:
                st.error(f"Model training failed: {e}")

        if "churn_metrics" in st.session_state:

            metrics = st.session_state["churn_metrics"]

            st.subheader("📊 Model Evaluation")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Accuracy",
                    f'{metrics["accuracy"]}%'
                )

            with col2:
                st.metric(
                    "Precision",
                    f'{metrics["precision"]}%'
                )

            col3, col4 = st.columns(2)

            with col3:
                st.metric(
                    "Recall",
                    f'{metrics["recall"]}%'
                )

            with col4:
                st.metric(
                    "F1 Score",
                    f'{metrics["f1_score"]}%'
                )

            st.write(
                "Training records:",
                metrics["training_rows"]
            )

            st.write(
                "Testing records:",
                metrics["testing_rows"]
            )

            st.subheader("Confusion Matrix")

            confusion_df = pd.DataFrame(
                metrics["confusion_matrix"],
                index=["Actual No Churn", "Actual Churn"],
                columns=["Predicted No Churn", "Predicted Churn"]
            )

            st.dataframe(
                confusion_df,
                use_container_width=True
            )
            
            # Individual Customer Churn Prediction

            st.divider()
            st.subheader("🔮 Individual Customer Churn Prediction")

            st.write(
                "Enter customer details to predict whether "
                "the customer is likely to churn."
            )

            model = st.session_state["churn_model"]
            original_df = st.session_state["uploaded_df"]

            # Get the features used by the trained model
            feature_columns = list(model.feature_names_in_)

            customer_data = {}

            with st.form("customer_prediction_form"):

                st.write("### Customer Information")

                col1, col2 = st.columns(2)

                for index, column in enumerate(feature_columns):

                    series = original_df[column]
                    clean_series = series.dropna()

                    if pd.api.types.is_numeric_dtype(series):

                        if len(clean_series) > 0:
                            default_value = float(clean_series.median())
                        else:
                            default_value = 0.0

                        if index % 2 == 0:
                            with col1:
                                customer_data[column] = st.number_input(
                                    column,
                                    value=default_value
                                )
                        else:
                            with col2:
                                customer_data[column] = st.number_input(
                                    column,
                                    value=default_value
                                )

                    else:

                        options = clean_series.astype(str).unique().tolist()

                        if not options:
                            options = ["Unknown"]

                        if index % 2 == 0:
                            with col1:
                                customer_data[column] = st.selectbox(
                                    column,
                                    options=options
                                )
                        else:
                            with col2:
                                customer_data[column] = st.selectbox(
                                    column,
                                    options=options
                                )

                predict_button = st.form_submit_button(
                    "🔍 Predict Customer Churn"
                )

            if predict_button:

                customer_df = pd.DataFrame(
                    [customer_data],
                    columns=feature_columns
                )

                try:
                    prediction = model.predict(customer_df)[0]

                    st.subheader("📊 Prediction Result")

                    if int(prediction) == 1:
                        st.error(
                            "⚠️ This customer is predicted to churn."
                        )
                    else:
                        st.success(
                            "✅ This customer is predicted to stay."
                        )

                    if hasattr(model, "predict_proba"):

                        probabilities = model.predict_proba(
                            customer_df
                        )[0]

                        classes = list(model.classes_)

                        if 1 in classes:
                            churn_probability = probabilities[
                                classes.index(1)
                            ] * 100

                            st.metric(
                                "Predicted Churn Probability",
                                f"{churn_probability:.2f}%"
                            )

                            if churn_probability >= 70:
                                st.error("Risk Level: High")
                            elif churn_probability >= 40:
                                st.warning("Risk Level: Medium")
                            else:
                                st.success("Risk Level: Low")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")



# ----------------------------------
# 7. AI INSIGHTS PAGE
# ----------------------------------

elif page == "AI Insights":

    st.header("🤖 AI-Powered Business Insights")

    st.write(
        "Ask questions about your customer dataset "
        "and get AI-powered business recommendations."
    )

    # Check whether dataset is available
    if "cleaned_df" in st.session_state:

        df = st.session_state["cleaned_df"]

    elif "uploaded_df" in st.session_state:

        df = st.session_state["uploaded_df"]

    else:

        st.warning(
            "Please upload a dataset from Dataset Analysis page first."
        )
        st.stop()

    st.subheader("Ask Gemini AI")

    user_question = st.text_input(
        "Ask a question about your dataset:",
        placeholder="Why are customers leaving?"
    )

    if st.button("Generate AI Insights"):

        if user_question.strip():

            with st.spinner("Gemini is analyzing your question..."):

                dataset_summary = df.describe(include="all").to_string()

                prompt = f"""
                You are a professional data analyst.

                Analyze the following customer dataset summary.

                Dataset Summary:
                {dataset_summary}

                User Question:
                {user_question}

                Provide:
                1. Clear explanation.
                2. Important observations.
                3. Business recommendations.

                Use simple language.
                """

                response = generate_ai_insights(prompt)

                st.subheader("Gemini AI Response")
                st.write(response)

        else:

            st.warning("Please enter a question first.")

