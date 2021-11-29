from seaborn import heatmap
from matplotlib import pyplot as plt
import pandas as pd


def main():
    x_train, y_train = pd.read_csv('x_train.csv'), pd.read_csv('y_train.csv')
    train_set = pd.concat([x_train, y_train], axis=1)
    train_set_corr = train_set.corr()
    ft_heatmap = heatmap(train_set_corr)
    plt.show()


if __name__ == "__main__":
    main()
