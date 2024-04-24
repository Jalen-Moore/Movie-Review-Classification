import numpy as np
import os
import re
import pickle

from sklearn.feature_extraction.text import HashingVectorizer

cur_dir = os.path.dirname(__file__)
stop = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'stopwords.pkl'), 'rb'))

def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)

    text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

vect = HashingVectorizer(decode_error='ignore', n_features=(2**21), preprocessor=None, tokenizer=tokenizer)

clf = pickle.load(open(os.path.join(cur_dir, 'pkl_objects', 'classifier.pkl'), 'rb'))

label = {0: 'negative', 1: 'positive'}

example = ["I love this movie. It's amazing."]
X = vect.transform(example)
# print("Prediction: ", label[clf.predict(X)[0]])
# print("Probability: ", np.max(clf.predict_proba(X))*100)