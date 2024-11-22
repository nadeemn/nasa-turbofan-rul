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
    fig = plt.figure(figsize=(15,10))
    plt.title('correlation')
    sns.heatmap(data, annot=True, fmt='0.2f', cmap='crest', linewidths=0.01)
    st.pyplot(fig)

def plot_line_chart(data, x_col, y_col, engines):
    fig = plt.figure(figsize=(15,10))
    for engine in engines:
        rolling_window = data[data['unit_number'] == engine].rolling(10).mean()
        sns.lineplot(data=rolling_window, x = x_col, y=y_col, label=engine)
    st.pyplot(fig)

def plot_histogram(data, chart, column, engines):
    fig = plt.figure(figsize=(15,10))

    n = len(engines)
    n_cols = 2
    n_rows = (n + n_cols-1)//n_cols

    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 5))
    ax = ax.flatten()

    for i, engine in enumerate(engines):
        filtered_data = data[data['unit_number'] == engine]
        sns.histplot(filtered_data[column], kde=True, ax= ax[i], label=f'Engine {engine}')
        ax[i].set_xlabel(column)
        ax[i].set_ylabel('Frequency')
        ax[i].set_ylabel(f'Distribution for Engine {engine}')
        ax[i].legend()

    for j in range(i+1, len(ax)):
        fig.delaxes(ax[j])

    plt.tight_layout()
    st.pyplot(fig)

def plot_boxplot(data, column, engines):
    figure = plt.figure(figsize=(15,10))

    for i, engine in enumerate(engines):
        filtered_data = data[data['unit_number'] == engine]
        sns.boxplot(x = filtered_data['unit_number'], y= filtered_data[column], width=0.3, showmeans=True)

    plt.xlabel('Engine type')
    plt.ylabel(column)

    plt.tight_layout()
    st.pyplot(figure)

def constant_features(df):
    constant_feature = []
    for col in df.columns:
        if abs(df[col].std() < 0.02):
            constant_feature.append(col)

    return constant_feature

def perform_action(df, to_remove_columns):
    return df.drop(columns=to_remove_columns)

def max_cycle(train_data):
    max = train_data[['unit_number', 'time']].groupby('unit_number').count().reset_index().rename(columns={'time': 'max cycle time'})
    return max