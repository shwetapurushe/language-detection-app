import pandas as pd
import numpy as np 
import re, os
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

data = pd.read_csv('Language_Detection.csv')
df = data.copy()

def dataClean(text):
    text = re.sub(r'[!@#$()\[\],\n"%^*?\:;~`0-9]', ' ', text)
    text = text.lower()
    return text

cleaned_text, cleaned_lang = df["Text"].apply(dataClean), df["Language"].apply(dataClean)
df = pd.DataFrame({'Text': cleaned_text, 'Language': cleaned_lang})

X = df["Text"]
y = df["Language"]

# Processing Y
l_encoder = LabelEncoder()
y_classes = l_encoder.fit_transform(y)

print(f"Encoded Langugages are * * *: {l_encoder.classes_}")

# We need to train test split before X processing
X_train, X_test , y_train , y_test = train_test_split(X, y_classes, test_size= 0.2, random_state=42)


cVect = CountVectorizer()
# # ML
mnb_model = MultinomialNB()

# # Using a pipeline
# ml_pipeline = Pipeline([('cVectorizer', cVect), ('multinomialNB', mnb_model)])
# ml_pipeline.fit(X_train, y_train)

# pipeline_preds = ml_pipeline.predict(X_test)
# acscore2 = accuracy_score(y_test, pipeline_preds)
# print(f"Pipeline Score is : {acscore2}")

# packaging the model
path = os.path.join(os.getcwd(), 'app', 'model', 'trained_language_detector-0.1.0.pkl')
# with open(path,'wb') as f:
#     pickle.dump(ml_pipeline, f)

# loading packaged model
deserialized_model = pickle.load(open(path, 'rb'))
y_preds = deserialized_model.predict(X_test)
print(f"Loaded model score is  * * * :", accuracy_score(y_test, y_preds))

# input = "Hello, my name is Shweta. What is your name?"
input = "Wie geht es ihnen?"
# my_y = ml_pipeline.predict([input]) # using pipeline directly
my_y = deserialized_model.predict([input]) # loading serialized model
print(f"For {input} : the language predicted is * * * {l_encoder.classes_[my_y[0]]}, at index {my_y}")


