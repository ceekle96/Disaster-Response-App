import sys
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import pickle

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, , AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

import warnings

warnings.simplefilter('ignore')

def load_data(database_filepath):
    """Load and merge messages and categories datasets

    Args:
    database_filename: string. Filename for SQLite database containing cleaned message data.

    Returns:
    X: dataframe. Dataframe containing features dataset.
    Y: dataframe. Dataframe containing labels dataset.
    category_names: list of strings. List containing category names.
    """
    # Load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM Messages", engine)

    # Create X and Y datasets
    X = df['message']
    Y = df.iloc[:, 4:]

    # Create list containing all category names
    category_names = list(Y.columns.values)

    return X, Y, category_names


def tokenize(text):
    """Normalize, tokenize and stem text string

    Args:
    text: string. String containing message for processing

    Returns:
    words: list of strings. List containing word tokens
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    words = word_tokenize(text)

    # remove stop words
    stopwords_ = stopwords.words("english")
    words = [word for word in words if word not in stopwords_]

    # extract root form of words
    words = [WordNetLemmatizer().lemmatize(word, pos='v') for word in words]

    return words


def build_model():
    """Build a machine learning pipeline

    Args:
    None

    Returns:
    cv: gridsearchcv object. Gridsearchcv object that transforms the data, creates the
    model object and finds the optimal model parameters.
    """
    # Create pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Create parameters dictionary
    parameters = {'vect__min_df': [1, 5],
                  'tfidf__use_idf': [True, False],
                  'clf__estimator__n_estimators': [10, 25],
                  'clf__estimator__min_samples_split': [2, 5, 10]}

    # Create grid search object
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Returns test accuracy, precision, recall and F1 score for fitted model

    Args:
    model: model object. Fitted model object.
    X_test: dataframe. Dataframe containing test features dataset.
    Y_test: dataframe. Dataframe containing test labels dataset.
    category_names: list of strings. List containing category names.

    Returns:
    None
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test.values, target_names=category_names))
    # print raw accuracy score
    print('Accuracy Score: {}'.format(np.mean(Y_test.values == y_pred)))


def save_model(model, model_filepath):
    """Pickle fitted model

    Args:
    model: model object. Fitted model object.
    model_filepath: string. Filepath for where fitted model should be saved

    Returns:
    None
    """
    pickle.dump(model.best_estimator_, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()