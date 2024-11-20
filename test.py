import os
import pandas as pd

import seaborn as sns
from kaggle_dataset import DatasetDownloader

column_names = ['unit_number', 'time', 'os_1', 'os_2', 'os_3']
sensor_measurements = [f'sm_{i}' for i in range(1, 22)]

column_names.extend(sensor_measurements)

downloader = DatasetDownloader("behrad3d/nasa-cmaps")
downloaded_path = downloader.download()

path = os.path.join(downloaded_path, 'CMaps', 'train_FD001.txt')
train_data = pd.read_csv(f'{path}', sep= ' ', header=None, names = column_names, index_col=False)
print(train_data.shape)
