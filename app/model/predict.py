import re , os
import pickle

def dataClean(text):
    text = re.sub(r'[!@#$()\[\],\n"%^*?\:;~`0-9]', ' ', text)
    text = text.lower()
    return text

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


# packaging the model
path = os.path.join(os.getcwd(), 'app', 'model', 'trained_language_detector-0.1.0.pkl')

model = pickle.load(open(path, 'rb'))

def predict_pipeline(text_input):
    cleaned_input = dataClean(text_input)
    pred_lang = model.predict([cleaned_input])
    return classes[pred_lang[0]]

