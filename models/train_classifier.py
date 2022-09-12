import sys

from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, make_scorer
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
import numpy as np
import joblib
nltk.download('wordnet')  # download for lemmatization
nltk.download('stopwords')
nltk.download('punkt')
from transformation import multi_class_score, tokenize

def get_metrics(test_value, predicted_value):
    """
    get_metrics calculates f1 score, accuracy and recall

    Args:
        test_value (list): list of actual values
        predicted_value (list): list of predicted values

    Returns:
        dictionray: a dictionary with accuracy, f1 score, precision and recall
    """
    accuracy = accuracy_score(test_value, predicted_value)
    precision = round(precision_score(
        test_value, predicted_value, average='micro'))
    recall = recall_score(test_value, predicted_value, average='micro')
    f1 = f1_score(test_value, predicted_value, average='micro')
    return {'Accuracy': accuracy, 'f1 score': f1, 'Precision': precision, 'Recall': recall}


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table("message_table", engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = list(Y.columns)
    return X, Y, category_names


def build_model():
    # write custom scoring for multiclass classifier
    # compute bag of word counts and tf-idf values
    vectorizer = TfidfVectorizer(
        tokenizer=tokenize, use_idf=True, smooth_idf=True, sublinear_tf=False)

    # clf = MultiOutputClassifier(RandomForestClassifier(random_state = 42))
    clf = RandomForestClassifier(random_state=42)

    pipeline = Pipeline([('vectorizer', vectorizer), ('clf', clf)])
    score = make_scorer(multi_class_score)
    parameters = {
        'clf__criterion': ['gini', 'entropy'],
        'clf__max_features': ['auto', 'sqrt'],
        'clf__max_depth': [5, 10, 20, 30, 40],
        'clf__random_state': [42]}

    cv_rf_tuned = GridSearchCV(pipeline, param_grid=parameters, scoring=score,
                               n_jobs=-1,
                               cv=5, refit=True, return_train_score=True, verbose=10)

    return cv_rf_tuned


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred_test = model.predict(X_test)

    test_results = []
    for i, column in enumerate(Y_test.columns):
        result = get_metrics(Y_test.loc[:, column].values, y_pred_test[:, i])
        test_results.append(result)
    test_results_df = pd.DataFrame(test_results)
    print("Result for Each Category")
    print(test_results_df)
    print("Overall Evaluation Result")
    print(test_results_df.mean())


def save_model(model, model_filepath):
    model = model.best_estimator_
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

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
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
