import matplotlib.pyplot as plt
import pandas as pd

def bar_plot(s: pd.Series, title:str, labels=None):
    """
        Plots fancier bars.

        ## Args
            - s (pd.Series): the numeric series to plot
            - title (str): the title of the plot

        ## Returns
            Displays the plot
    """
    if labels is None:
        labels = []

    s.plot(kind='bar',figsize=(7, 5))
    plt.title(title)

    for i in range(len(s)):
        # Placing text at half the bar height
        plt.text(i, s.values[i] + 15, s.values[i], ha='center')

    if len(labels) != 0:
        plt.xticks([False, True], labels, rotation=90)