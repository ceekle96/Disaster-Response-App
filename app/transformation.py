
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import  accuracy_score
import numpy as np
def tokenize(text):
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    # Reduce words to their stems
    words = [PorterStemmer().stem(w) for w in words]
    words = [WordNetLemmatizer().lemmatize(w) for w in words]
    return words


def multi_class_score(y_true, y_pred):
    accuracy_results = []
    for i, column in enumerate(y_true.columns):
        accuracy = accuracy_score(
            y_true.loc[:, column].values, y_pred[:, i])
        accuracy_results.append(accuracy)
    avg_accuracy = np.mean(accuracy_results)
    return avg_accuracy