import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from kaggle_dataset import DatasetDownloader
from sklearn.preprocessing import MinMaxScaler
from utils import draw_distribution_chart, plot_correlation, plot_line_chart, plot_histogram, plot_boxplot, constant_features, perform_action


st.title("NASA Jet Engine Data Dashboard")
st.sidebar.title('Options')

uploaded_file = st.sidebar.file_uploader("Upload your dataset", type=["CSV", "TXT"])

column_names = ['unit_number', 'time', 'os_1', 'os_2', 'os_3']
sensor_measurements = [f'sm_{i}' for i in range(1, 22)]

column_names.extend(sensor_measurements)

if uploaded_file:
    train_data = pd.read_csv(uploaded_file, sep=' ', header=None, names= column_names, index_col=False)

    st.write("**Uploaded data**")
    st.dataframe(train_data)

    tab1, tab2, tab3 = st.tabs(["Exploratory Data Analysis", "Pre-Processing", "Visualization"])

    with tab1:
        subtab1, subtab2, subtab3, subtab4, subtab5, subtab6 = st.tabs(['Data Shape', "Data Types",
                                                    "Missing Values", "Statistical Summary",
                                                      "Number of Unique Values", "Compute the RUL"])
        with subtab1:
            st.write("**Shape of the dataset**", train_data.shape)
        with subtab2:
            st.write("**Data Types:**")
            st.write(train_data.dtypes)
        with subtab3:
            st.write("**Missing Values:**")
            st.write(train_data.isnull().sum())
        with subtab4:
            st.write("**Statistical Summary:**")
            st.write(train_data.describe().transpose())
        with subtab5:
            st.write("**Number of Unique values**")
            st.write(train_data.nunique())
        with subtab6:
            st.write("### Compute Remaining Useful Life (RUL)")
            time_column = st.selectbox("Select the time column", train_data.columns)

            max_cycle_value = train_data.groupby('unit_number')['time'].transform('max')

            train_data['RUL'] = max_cycle_value - train_data[time_column]
            st.write("RUL Column Added")
            st.dataframe(train_data.head())

            def max_cycle(train_data):
                max = train_data[['unit_number', 'time']].groupby('unit_number').count().reset_index().rename(columns={'time': 'max cycle time'})
                return max
            st.write("**Max cycle of each Engine**")
            max_engine_cycle = max_cycle(train_data)
            st.write(max_engine_cycle)

            st.bar_chart(data=max_engine_cycle, x='unit_number', y='max cycle time')

            draw_distribution_chart(max_engine_cycle['max cycle time'])

    with tab2:
        st.header("Preprocessing")
        df = train_data.copy()

        subtab2_1, subtab2_2, subtab2_3 = st.tabs(["Handle Missing Values", "Feature Selection", "Feature Normalization"])

        with subtab2_1:
            total_missing_values = df.isnull().sum()
            if total_missing_values.sum() == 0:
                st.write("**NO Missing value found.**")

        with subtab2_2:
            st.write("**Select the columns that need to be droped.**")
            constant_feuture_list = constant_features(df)

            selected_features = st.multiselect("Select the column",
                            options=[col for col in df.columns if col not in ["RUL"]],
                              default=constant_feuture_list, key="featurelistoptions")
            
            if st.button("Drop"):
                result_data = perform_action(df, selected_features)

                st.write("DataFrame after Removing")
                st.dataframe(result_data)
                df = result_data.copy()

        with subtab2_3:
            st.write("**Chose normalization technique**")

            min_max = st.checkbox("Min-Max Normalization", value=False)

            if min_max:
                scaler = MinMaxScaler()
                scaled_data = scaler.fit_transform(df)
                scaled_data = pd.DataFrame(scaled_data, columns=df.columns)

                st.write("Scaled Data")
                st.dataframe(scaled_data)
        
    with tab3:
        subtab1_3, subtab2_3, subtab3_3 = st.tabs(["Correlation Matrix", "Line Chart", "Data Distribution" ])
        with subtab1_3:
            df_corr = train_data.corr()
            mask = np.tril(np.ones(df_corr.shape), k=-1).astype(bool)
            df_corr = df_corr.where(mask)

            plot_correlation(df_corr)

        with subtab2_3:
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

        with subtab3_3:
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

