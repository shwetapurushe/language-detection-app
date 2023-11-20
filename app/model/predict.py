import re , os
import pickle
from app.utils.utils import dataClean, loadTrainedModel, MODELNAME

classes = [
    "Arabic",
    "Danish",
    "Dutch",
    "English",
    "French",
    "German",
    "Greek",
    "Hindi",
    "Italian",
    "Kannada",
    "Malayalam",
    "Portugeese",
    "Russian",
    "Spanish",
    "Sweedish",
    "Tamil",
    "Turkish",
]


# loading the trained model
model = loadTrainedModel(MODELNAME)

def predict_pipeline(text_input):
    cleaned_input = dataClean(text_input)
    pred_lang = model.predict([cleaned_input])
    return classes[pred_lang[0]]

