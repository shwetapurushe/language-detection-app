import re, os
import pickle

MODELNAME = 'trained_language_detector-0.1.0.pkl'

def dataClean(text):
    text = re.sub(r'[!@#$()\[\],\n"%^*?\:;~`0-9]', ' ', text)
    text = text.lower()
    return text

def loadTrainedModel(modelName):
    path = os.path.join(os.getcwd(), 'app', 'model', modelName)
    return pickle.load(open(path, 'rb'))
