import sys
sys.path.insert(1, '../main')

import pickle
import utility
from common import extract_data
from mlmodel import NN
from utility import get_feature_vector_from_mfcc
from record import record_audio

while (True):
    prompt = input("Enter r to record or q to quit: ")

    if (prompt=='r'):
        record_audio()
        filename = 'finalized_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        #These have been removed because they are only used to train the model, but it is already trained
        #x_train, x_test, y_train, y_test, _ = extract_data(flatten=True)
        #loaded_model.evaluate(x_test, y_test)
        filename = "../AudioData/testaudio.wav"
        print(loaded_model.predict_one(get_feature_vector_from_mfcc(filename, flatten=True)))
    elif (prompt=='q'):
        sys.exit()
    else:
        print("Error, Invalid response. Please try again: ")