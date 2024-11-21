import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Set up the app
st.title("EDA & Preprocessing Dashboard")
st.sidebar.title("Options")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv","txt"])

if uploaded_file:
    # Read the file
    data = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(data.head())

    # Sidebar options
    show_summary = st.sidebar.checkbox("Show Summary Statistics", value=True)
    visualize_data = st.sidebar.checkbox("Visualize Data", value=True)
    handle_missing = st.sidebar.checkbox("Handle Missing Values", value=False)
    encode_data = st.sidebar.checkbox("Encode Categorical Data", value=False)
    scale_data = st.sidebar.checkbox("Scale Numerical Data", value=False)

    # Summary statistics
    if show_summary:
        st.write("### Summary Statistics")
        st.write("**Shape of the dataset:**", data.shape)
        st.write("**Data Types:**")
        st.write(data.dtypes)
        st.write("**Missing Values:**")
        st.write(data.isnull().sum())
        st.write("**Statistical Summary:**")
        st.write(data.describe())

    # Data visualization
    if visualize_data:
        st.write("### Data Visualization")
        st.write("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("#### Distribution of Numerical Columns")
        numeric_columns = data.select_dtypes(include=np.number).columns
        for col in numeric_columns:
            st.write(f"Distribution for `{col}`:")
            fig, ax = plt.subplots()
            sns.histplot(data[col], kde=True, ax=ax)
            st.pyplot(fig)

    # Missing value handling
    if handle_missing:
        st.write("### Handle Missing Values")
        missing_strategy = st.selectbox(
            "Select strategy to handle missing values",
            options=["Drop rows", "Fill with mean", "Fill with median", "Fill with mode"],
        )
        if missing_strategy == "Drop rows":
            data = data.dropna()
            st.write("Rows with missing values dropped.")
        elif missing_strategy == "Fill with mean":
            data = data.fillna(data.mean())
            st.write("Missing values filled with mean.")
        elif missing_strategy == "Fill with median":
            data = data.fillna(data.median())
            st.write("Missing values filled with median.")
        elif missing_strategy == "Fill with mode":
            data = data.fillna(data.mode().iloc[0])
            st.write("Missing values filled with mode.")
        st.write("Updated Dataset:")
        st.dataframe(data.head())

    # Categorical encoding
    if encode_data:
        st.write("### Encode Categorical Data")
        categorical_columns = data.select_dtypes(include=['object', 'category']).columns
        if len(categorical_columns) > 0:
            st.write(f"Categorical columns: {list(categorical_columns)}")
            encoder = LabelEncoder()
            for col in categorical_columns:
                data[col] = encoder.fit_transform(data[col])
            st.write("Categorical data encoded using Label Encoding.")
        else:
            st.write("No categorical columns to encode.")
        st.dataframe(data.head())

    # Scaling numerical data
    if scale_data:
        st.write("### Scale Numerical Data")
        numeric_columns = data.select_dtypes(include=np.number).columns
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
            st.write("Numerical data scaled using Standard Scaler.")
        else:
            st.write("No numerical columns to scale.")
        st.dataframe(data.head())

    # Download preprocessed data
    st.write("### Download Preprocessed Data")
    csv = data.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "preprocessed_data.csv", "text/csv")
else:
    st.write("Upload a CSV file to get started.")
