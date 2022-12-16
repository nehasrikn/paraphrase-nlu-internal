import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Any
import pandas as pd

def plot_and_save_countplot(values: List[Any], fig_file: str) -> None:
    sns.set_theme()
    plt.figure()
    plot = sns.countplot(data=pd.DataFrame(values, columns=['value']), x="value")
    plt.xticks(rotation=45, ha='right')
    plot.get_figure().savefig(fig_file)

def plot_and_save_hist(values: List[Any], fig_file: str) -> None:
    sns.set_theme()
    plt.figure()
    plot = sns.histplot(data=pd.DataFrame(values, columns=['value']), x="value", kde=True)
    plot.get_figure().savefig(fig_file)