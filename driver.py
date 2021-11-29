# Driver file.
# Used for conveniently sandboxing and comparing different models' performance.
from lgr import process_lgr
from nb import process_nb
from rf import process_rf
from dt import process_dt
from time import time


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
    return available_models[model]()


def main():
    models = dict()
    for model in ['dt', 'nb', 'lgr', 'rf']:
        print("Training {} model...".format(model))
        start_time = time()
        models[model] = train_model(model)
        print("Training and prediction completed in {}s.".format(round(time() - start_time), 4))

    # Models are saved in the dictionary models and can be accessed through models[model_name]


main()
