

# load all csv files in results folder
import os
from pathlib import Path

import pandas as pd
import numpy as np
# use seaborn for nice plots
import seaborn as sns

import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go


# current dir path
current_dir = Path(__file__).resolve().parent
results_path = current_dir / 'results'

def visualize_preds():
    csv_files = results_path.glob('predictions-2022-01-08*.csv')

    # sort csv files by creation date
    csv_files = sorted(csv_files, key=lambda x: x.stat().st_ctime)
    file_names = [x.stem for x in csv_files]
    dates_from_file_names = ['-'.join(x.split('-')[1:]) for x in file_names]
    # read date from dates formatted as YYYY-MM-DD_HH-MM-SS
    # 2022-01-07_08-59-04
    dates_from_file_names = [pd.to_datetime(x, format='%Y-%m-%d_%H-%M-%S') for x in dates_from_file_names]
    a_preds = []
    a_names = []
    b_preds = []
    b_names = []
    i=0
    for csv_file, name in zip(csv_files, dates_from_file_names):
        print(csv_file)

        with open(csv_file, 'r') as f:
            df = pd.read_csv(f)
            # get row where instrument == TSLA
            df = df[df['instrument'] == 'TSLA']
            i+=1
            if i % 2 == 0:
                a_preds.append(df['close_predicted_price'])
                a_names.append(name)
            else:
                b_preds.append(df['close_predicted_price'])
                b_names.append(name)
            # all_preds.append(df['close_predicted_price'])
    ## plot all preds
    a_preds = np.array(a_preds)
    a_preds = a_preds.reshape(a_preds.shape[0])
    b_preds = np.array(b_preds)
    b_preds = b_preds.reshape(b_preds.shape[0])

    # sns.set(style="darkgrid")
    # sns.lineplot(x=np.arange(a_preds.shape[0]), y=a_preds)
    # really large size plot
    # plt.gcf().set_size_inches(10, 10)
    # #high quality plot
    # plt.savefig('predictions.png', dpi=300)
    # plt.show()


    # plotly graph looks better
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=a_names, y=a_preds))
    plotly.offline.plot(fig, filename='predictions.html')

    # plotly graph looks better
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=b_names, y=b_preds))
    plotly.offline.plot(fig, filename='predictions_b.html')
    # plt.plot(a_preds)
    # plt.show()
    print(a_preds)

if __name__ == '__main__':
    visualize_preds()
