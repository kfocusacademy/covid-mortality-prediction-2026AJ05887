import streamlit as st
import data_model
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="COVID-19 Mortality Prediction",
    page_icon="🏥",
    layout="wide"
)

# Application title
st.title("COVID-19 Patient Mortality Prediction")

st.write(
    "Machine Learning application to evaluate "
    "COVID-19 mortality prediction models."
)

# Dataset selection
st.subheader("Dataset Selection")

dataset_option = st.radio(
    "Choose your dataset source:",
    ["Upload CSV", "Use default test_data.csv"],
    horizontal=True
)
uploaded_file = None
df = None

if dataset_option == "Upload CSV":

    uploaded_file = st.file_uploader(
        "Upload your COVID-19 dataset",
        type=["csv"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")

else:
    df = pd.read_csv("covid_test_data.csv")
    st.success("Default test dataset loaded successfully!")

if df is not None:
    model_files = {
    "Logistic": "logistic_regression.joblib",
    "DecisionTree": "decision_tree.joblib",
    "SVM": "svm.joblib",
    "KNN": "knn.joblib",
    "NaiveBayes": "naivebayes.joblib"
    }
    
    # test_data.csv already contains the encoded target
    test_y = df["DATE_DIED"]
    test_x = df.drop(columns=["DATE_DIED"])
    st.write("Test dataset shape:", test_x.shape)

    st.write("Dataset preparation completed!")

    # Display dataset information
    st.write("Testing data shape:", test_x.shape)

    st.write("Testing target shape:", test_y.shape)
    st.subheader("Model Selection")

    model_options = [
        "Logistic",
        "DecisionTree",
        "SVM",
        "KNN",
        "NaiveBayes"]

    selected_model = st.selectbox(
        "Choose a Machine Learning Model",
        model_options
    )

    st.write("Selected Model:", selected_model)

    if st.button("Run Model"):

        data_model.results.clear()
        data_model.classification_reports.clear()
        data_model.confusion_matrices.clear()
        
        with st.spinner("Training model and evaluating performance..."):

            model = data_model.load_model(
                model_files[selected_model]
                )

            y_pred = model.predict(test_x)
            y_prob = model.predict_proba(test_x)

            death_index = list(model.classes_).index(1)

            data_model.evaluate_metrics(
                selected_model + "_FineTuned",
                test_y,
                y_pred,
                y_prob,
                death_index
            )

        st.success("Model evaluation completed!")
    
        st.subheader("Model Performance Comparison")
        results_df = data_model.print_results() 
        st.dataframe(results_df, use_container_width=True)
        
        # Displaying classification report
        st.subheader("Classification Reports")
        for item in data_model.classification_reports:
            st.write(item["Model"])
            st.dataframe(
                pd.DataFrame(item["Report"]).transpose()
            )

        # Displaying confusion matrix
        st.subheader("Confusion Matrices")
    
        for item in data_model.confusion_matrices:
            st.write(item["Model"])
            st.dataframe(
                pd.DataFrame(
                    item["Matrix"],
                    index=["Actual Died", "Actual Survived"],
                    columns=["Predicted Died", "Predicted Survived"]
                )
                )
    
    if st.button("Run All Models"):
        data_model.results.clear()
        data_model.classification_reports.clear()
        data_model.confusion_matrices.clear()
        with st.spinner("Training and evaluating all models..."):

            for model_name, filename in model_files.items():
                st.write(f"Running {model_name}...")
                model = data_model.load_model(filename)
                y_pred = model.predict(test_x)
                y_prob = model.predict_proba(test_x)

                death_index = list(model.classes_).index(1)

                data_model.evaluate_metrics(
                            model_name + "_FineTuned",
                            test_y,
                            y_pred,
                            y_prob,
                            death_index
                            )
                
        st.success("All models executed successfully!")
        st.subheader("Overall Model Performance")
        results_df = data_model.print_results()
        st.dataframe(results_df, use_container_width=True)
        # Displaying classification report
        st.subheader("Classification Reports")
        for item in data_model.classification_reports:
            st.write(item["Model"])
            st.dataframe(
                pd.DataFrame(item["Report"]).transpose()
            )

        # Displaying confusion matrix
        st.subheader("Confusion Matrices")

        for item in data_model.confusion_matrices:
            st.write(item["Model"])
            st.dataframe(
                pd.DataFrame(
                    item["Matrix"],
                    index=["Actual Died", "Actual Survived"],
                    columns=["Predicted Died", "Predicted Survived"]
                )
            )