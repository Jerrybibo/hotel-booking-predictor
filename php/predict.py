# For usage in web server. See more at http://jerry.games/334

import sys
import base64
import json
import pickle

# Obtain the input and convert to dict
arg = sys.argv[1]
content = json.loads(base64.b64decode(arg))

# Unpickle our classifier of choice
classifier_name = content['classifier']
with open('{}.pickle'.format(model)) as f:
    classifier = pickle.load(f)

# Todo preprocess our input data to match that requested by the models
