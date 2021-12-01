# Driver file.
# Used for conveniently sandboxing and comparing different models' performance.

import pickle
from clf_models.lgr import process_lgr
from clf_models.nb import process_nb
from clf_models.rf import process_rf
from clf_models.dt import process_dt
from time import time
from pandas import read_csv
from globals import USE_LEGACY_DATASET

if USE_LEGACY_DATASET:
    x_train = read_csv('legacy/xTrain.csv').to_numpy()
    x_test = read_csv('legacy/xTest.csv').to_numpy()
    y_train = read_csv('legacy/yTrain.csv').to_numpy()
    y_test = read_csv('legacy/yTest.csv').to_numpy()
else:
    x_train = read_csv('x_train.csv').to_numpy()
    x_test = read_csv('x_test.csv').to_numpy()
    y_train = read_csv('y_train.csv').to_numpy()
    y_test = read_csv('y_test.csv').to_numpy()


def train_model(model):
    available_models = {
        'dt': process_dt,
        'nb': process_nb,
        'rf': process_rf,
        'lgr': process_lgr
    }
    if model not in available_models.keys():
        raise Exception("The specified model {} is not part of the available models ({}).".format(
            model, available_models.keys()
        ))
    return available_models[model](x_train, x_test, y_train, y_test)


def main():
    models = dict()
    for model in ['dt', 'nb', 'lgr', 'rf']:
        print("Training {} model...".format(model))
        start_time = time()
        models[model] = train_model(model)
        print("Training and prediction completed in {}s.".format(round(time() - start_time, 4)))

    # Models are saved in the dictionary models and can be accessed through models[model_name]
    # Pickle the models to eliminate need for training every time we run
    for key, model in models.items():
        with open('pickled_models/{}.pickle'.format(key), 'wb') as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
        print('pickled_models/{}.pickle'.format(key), "file created")


if __name__ == "__main__":
    main()
