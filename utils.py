import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def draw_distribution_chart(data):
    fig, ax = plt.subplots()
    sns.histplot(data, kde='True')
    plt.tight_layout()
    plt.xlabel('cycle')
    plt.ylabel('frequency')
    st.pyplot(fig)

def plot_correlation(data):
    fig = plt.figure(figsize=(15,7))
    plt.title('correlation')
    sns.heatmap(data, annot=True, fmt='0.2f', cmap='crest', linewidths=0.01)
    st.pyplot(fig)

def plot_line_chart(data, x_col, y_col, engines):
    fig = plt.figure(figsize=(12,7))
    for engine in engines:
        rolling_window = data[data['unit_number'] == engine].rolling(10).mean()
        sns.lineplot(data=rolling_window, x = x_col, y=y_col, label=engine)
    st.pyplot(fig)

def plot_histogram(data, chart, column, engines):
    fig = plt.figure(figsize=(12,7))
    filtered_data = data[data['unit_number'].isin(engines)]
    st.write(filtered_data)
    sns.histplot(filtered_data[column], kde=True, multiple="stack", palette="Set2")
    plt.tight_layout()
    plt.xlabel(column)
    plt.ylabel('Frequency')
    st.pyplot(fig)