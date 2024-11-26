import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from kaggle_dataset import DatasetDownloader
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score

from utils import draw_distribution_chart, plot_correlation, plot_line_chart, plot_histogram, plot_boxplot, constant_features, perform_action, max_cycle

st.title("NASA Jet Engine Data Dashboard")
st.sidebar.title('Options')

uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["CSV", "TXT"])

column_names = ['unit_number', 'time', 'os_1', 'os_2', 'os_3']
sensor_measurements = [f'sm_{i}' for i in range(1, 22)]

column_names.extend(sensor_measurements)

if "training_ready" not in st.session_state:
    st.session_state.training_ready = False


if not uploaded_file:
 
    st.header("Prediction Goal")
    st.write("""
    In this dataset, the goal is to predict the Remaining Useful Life (RUL) of each engine in the test dataset. 
    RUL is equivalent to the number of flights remaining for the engine after the last data point in the test dataset.
    """)

    st.header("Experimental Scenario")
    st.write("""
    The data set consists of multiple multivariate time series. Each data set is further divided into training and test subsets. 
    Each time series is from a different engine, i.e., the data can be considered to be from a fleet of engines of the same type. 
    Each engine starts with different degrees of initial wear and manufacturing variation, which is unknown to the user. 
    This wear and variation is considered normal and not a fault condition.

    There are three operational settings that have a substantial effect on engine performance. These settings are also included in the data. 
    The data is contaminated with sensor noise.

    The engine is operating normally at the start of each time series and develops a fault at some point during the series. 
    In the training set, the fault grows in magnitude until system failure. In the test set, the time series ends sometime prior to system failure. 
    The objective is to predict the number of remaining operational cycles before failure in the test set, i.e., the number of operational cycles 
    after the last cycle that the engine will continue to operate. A vector of true Remaining Useful Life (RUL) values for the test data is also provided.
    """)

    st.header("Data Details")
    st.write("""
    The data are provided as a zip-compressed text file with 26 columns of numbers, separated by spaces. 
    Each row is a snapshot of data taken during a single operational cycle, and each column is a different variable.
    """)

    st.header("Dataset Link")
    st.markdown(
        "You can [download the dataset here](https://www.kaggle.com/datasets/behrad3d/nasa-cmaps/data)."
    )


if uploaded_file:
    train_data = pd.read_csv(uploaded_file, sep=' ', header=None, names= column_names, index_col=False)

    dashboard_tab, model_tab = st.tabs(["Dashboard", "Model Training and Evaluation"])

    with dashboard_tab:

        st.write("**Uploaded data**")
        st.dataframe(train_data)

        eda_tab, visual_tab, preprocess_tab = st.tabs(["Exploratory Data Analysis", "Visualization", "Pre-Processing"])

        with eda_tab:
            datashape_tab, datatype_tab, missingValue_tab, statistics_tab, uniqueValue_tab, rul_tab = st.tabs(['Data Shape', "Data Types",
                                                    "Missing Values", "Statistical Summary",
                                                      "Number of Unique Values", "Compute the RUL"])
            with datashape_tab:
                st.write("**Shape of the dataset**", train_data.shape)
            with datatype_tab:
                st.write("**Data Types:**")
                st.write(train_data.dtypes)
            with missingValue_tab:
                st.write("**Missing Values:**")
                st.write(train_data.isnull().sum())
            with statistics_tab:
                st.write("**Statistical Summary:**")
                st.write(train_data.describe().transpose())
            with uniqueValue_tab:
                st.write("**Number of Unique values**")
                st.write(train_data.nunique())
            with rul_tab:
                st.write("### Compute Remaining Useful Life (RUL)")
                time_column = st.selectbox("Select the time/cycle column", train_data.columns)

                max_cycle_value = train_data.groupby('unit_number')['time'].transform('max')
                train_data['RUL'] = max_cycle_value - train_data[time_column]

                st.write("Remaining Useful cycle (RUL) Column Added")
                st.dataframe(train_data.head())

                if st.button("Show max cycle of each engine"):
                    st.write("**Max cycle of each Engine**")
                    max_engine_cycle = max_cycle(train_data)
                    st.write(max_engine_cycle)

                    st.bar_chart(data=max_engine_cycle, x='unit_number', y='max cycle time')

                    draw_distribution_chart(max_engine_cycle['max cycle time'])

        with preprocess_tab:
            st.header("Preprocessing")

            if "processed_data" not in st.session_state:
                st.session_state.processed_data = train_data.copy()

            df = st.session_state.processed_data

            handleMissingValues_tab, featureSelection_tab, featureNormalizationTab = st.tabs(["Handle Missing Values", "Feature Selection", "Feature Normalization"])

            with handleMissingValues_tab:
                total_missing_values = df.isnull().sum()
                if total_missing_values.sum() == 0:
                    st.write("**No Missing value found.**")

            with featureSelection_tab:
                st.write("**Select the columns that need to be droped.**")
                constant_feuture_list = constant_features(df)
                
                selected_features = st.multiselect("Select the column",
                                options= df.columns,
                                default=constant_feuture_list,
                                key="featurelistoptions")
                
                if st.button("Drop Features"):

                    if "features_selected" not in st.session_state:
                        st.session_state.features_selected = selected_features

                    st.session_state.processed_data = perform_action(df, selected_features)

                    st.write("DataFrame after Removing")
                    st.success("Selected features have been dropped")
                    st.dataframe(st.session_state.processed_data)

            with featureNormalizationTab:
                st.write("**Chose normalization technique**")

                normalization_method = st.radio(
                                            "Select a normalization technique", 
                                                options=["None", "Min-Max Normalization", "Standard Scaler Normalization"], 
                                                index=0
                                                )

                if normalization_method == "Min-Max Normalization":
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(df)
                    st.session_state.scaler = scaler
                    scaled_data = pd.DataFrame(scaled_data, columns=df.columns)

                    st.write("After Normalization")
                    st.dataframe(scaled_data)

                elif normalization_method == "Standard Scaler Normalization":
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df)
                    st.session_state.scaler = scaler
                    scaled_data = pd.DataFrame(scaled_data, columns=df.columns)

                    st.write("After Normalization")
                    st.dataframe(scaled_data)
                
                if normalization_method != "None":
                    if st.button("Prepare Training Dataset"):
                        st.session_state.training_ready = True
                        st.success("Training dataset Prepared! Navigate to the Model Training tab.")
        
        with visual_tab:
            correlation_tab, lineChart_tab, data_tab = st.tabs(["Correlation Matrix", "Line Chart", "Data Distribution" ])
            with correlation_tab:
                df_corr = train_data.corr()
                mask = np.tril(np.ones(df_corr.shape), k=-1).astype(bool)
                df_corr = df_corr.where(mask)

                plot_correlation(df_corr)

            with lineChart_tab:
                x_col = st.selectbox("Select the x-axis column", [col for col in train_data.columns if (col == "RUL" or col =="time")])
                y_col = st.selectbox("Select the y-axis column", [col for col in train_data.columns if col!=x_col and col!="unit_number" and col!="time"])
                engine_options = train_data['unit_number'].unique()
                selected_engines = st.multiselect(
                    "Select Engine numbers to Filter",
                    options=engine_options,
                    default=engine_options[:5]
                )
                if selected_engines:
                    plot_line_chart(train_data, x_col, y_col, selected_engines)
                else:
                    st.warning("Please select at least one engine to generate the plot.")

            with data_tab:
                col1, col2 = st.columns(2)
                with col1:
                    plot_name = st.selectbox("Select Plot", ["Histogram", "Boxplot"])
                with col2:
                    col_name = st.selectbox("Select column to plot", [col for col in train_data.columns if col not in ["unit_number", "time"]])
                engine_options = train_data['unit_number'].unique()
                selected_engines = st.multiselect(
                    "Select Engine numbers to Filter",
                    options=engine_options,
                    default=engine_options[:3],
                    key="plotdistribution"
                )
                if plot_name == "Histogram":
                    plot_histogram(train_data, plot_name, col_name, selected_engines)
                elif plot_name == "Boxplot":
                    plot_boxplot(train_data, col_name, selected_engines)
    
    with model_tab:
        if st.session_state.training_ready:

            traintab, valtab, testtab = st.tabs(["Train", "Validate", "Test"])

            with traintab:
            
                st.subheader("Train the model.")

                column1, column2 = st.columns(2)

                with column1:
                    model_choice = st.selectbox("Select the model:", ["Linear Regression", "SVM"])
            
                with column2:
                    test_size = st.slider("Select Validation size", 0.1, 0.2, 0.3, step=0.05)
            
                if st.button("Split Data and Train"):
                    X = st.session_state.processed_data
                    y = train_data['RUL']

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

                    st.write(f"Training data: {X_train.shape, y_train.shape}")
                    st.write(f"Test data: {X_test.shape, y_test.shape}")
                    st.session_state.data_split = True

                    if "data_split" in st.session_state:
                        st.session_state.x_test = X_test
                        st.session_state.y_test = y_test

                    st.write("Training in progress...")
                    
                    if model_choice == "Linear Regression":
                        model = LinearRegression()
                    elif model_choice == "SVM":
                        kernel_choice = "poly"
                        model = SVR(kernel=kernel_choice)

                    model.fit(X_train, y_train)
                    if "model_trained" not in st.session_state:
                        st.session_state.model_trained = model
                    st.success("model successfully trained")

            with valtab:
                if "model_trained" not in st.session_state:
                    st.write("You cant Evaluate the model before training.")

                else:
                    model = st.session_state.model_trained
                    if st.button("Evaluate on Validation Data"):

                        st.write("**Evaluating on Val data...**")
                        y_pred = model.predict(st.session_state.x_test)

                        mse = mean_squared_error(st.session_state.y_test, y_pred)
                        r2 = r2_score(st.session_state.y_test, y_pred)

                        st.write(f"Mean Squared Error {mse: .4f}")
                        st.write(f"R squared score: {r2: .4f}")

                        fig, ax = plt.subplots()
                        ax.scatter(st.session_state.y_test, y_pred, alpha=0.7)
                        ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()], [st.session_state.y_test.min(), st.session_state.y_test.max()], 'k--', lw=2)
                        ax.set_xlabel("True Values")
                        ax.set_ylabel("Predicted Values")
                        ax.set_title(f"{model_choice} Predictions vs True Values")
                        st.pyplot(fig)

            with testtab:
                if "model_trained" not in st.session_state:
                    st.write("You cant test the model before training.")
                else:
                    uploaded_test_data = st.file_uploader("upload test data")

                    if uploaded_test_data is not None:
                        test_data = pd.read_csv(uploaded_test_data,sep=' ', header=None, names= column_names, index_col=False)
                        st.dataframe(test_data)

                        exclude_column = "RUL"

                        selected_features = [col for col in st.session_state.features_selected if col!= exclude_column]
                        grouped_data = test_data.groupby("unit_number").last().reset_index()

                        if st.button("Preprocess the test data and predict"):
                            grouped_Data = grouped_data.drop(columns=selected_features)

                            if hasattr(st.session_state, "scaler"):
                                scaler = st.session_state.scaler

                                scaled_test_data = scaler.transform(grouped_Data)
                                st.dataframe(scaled_test_data)

                                y_pred_test = st.session_state.model_trained.predict(scaled_test_data)

                                st.write("prediction")
                                st.write(pd.DataFrame(y_pred_test, columns=["predicted values"]))
                    

        else:
            st.write("Dataset Not yet ready for training!")






