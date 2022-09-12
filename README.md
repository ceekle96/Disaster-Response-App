# Emergency Response Pipeline

# Table of Contents

1. [Instructions:](#instructions)
2. [Summary](#summary)
3. [File Description](#file-desc)
4. [Dataset](#data)
5. [Modeling Process](#model)
6. [Screenshots](#screenshots)
7. [Model results](#result)
8. [Effect of Imbalance](#effect)

### Instructions: <a name="instructions"></a>

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in the database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to http://0.0.0.0:3001/

### Summary: <a name="summary"></a>

In this project, I've analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.

The data set contains real messages that were sent during disaster events. I've created a machine learning pipeline to categorize these events so that messages can be sent to an appropriate disaster relief agency.

This project has an app inside the app folder. Using it an emergency worker can input a new message and get classification results in several categories. The web app also display the visualization of the data.

## File Description <a name="file-desc"></a>

- [**ETL Pipeline Preparation.ipynb**](notebooks/ETL%20Pipeline%20Preparation.ipynb):
  Notebook contains ETL Pipeline.
- [**ML Pipeline Preparation.ipynb**](notebooks/ML%20Pipeline%20Preparation.ipynb):
  Notebook contains ML Pipeline.
- [**etl.db**](notebooks/etl.db): etl database.
- [**categories.csv**](notebooks/categories.csv): Categories data set.
- [**messages.csv**](notebooks/messages.csv): Messages data set.
- [**classifier.pkl**](models/classifier.pkl): Trained model pickle file.
- [**train_classifier.py**](models/train_classifier.py): Python file for model training.
- [**transformation.py**](models/transformation.py): Helper file for train_classifier.py
- [**disaster_categories.csv**](data/disaster_categories.csv): Disaster Categories data set.
- [**disaster_messages.csv**](data/disaster_messages.csv): Disaster Messages data set.
- [**process_data.py**](data/process_data.py): Python ETL script.
- [**app**](app/): Flask Web App
- [**run.py**](app/): Flask Web App main script.
- [**img**](img/): Image Folder
- [**requirements.txt**](/requirements.txt): Text file containing list of packages used.
- [**LICENSE**](/LICENSE): Project LICENSE file.

### Dataset <a name="data"></a>

This disaster data is from [Figure Eight](https://www.figure-eight.com/)
This dataset has two files messages.csv and categories.csv.

### Data Cleaning

1. Based on id two datasets were first merged into df.
2. Categories were split into separate category columns.
3. Category values were converted to numbers 0 or 1.
4. Replaced categories column in df with new category columns.
5. Removed duplicates based on the message column.
6. df were exported to etl.db database.

### Modeling Process <a name="model"></a>

1. Wrote a tokenization function to process text data.
2. Build a machine learning pipeline using TfidfVectorizer, RandomForestClassifier, and Pipeline.
3. Split the data into training and test sets.
4. Using pipeline trained and evaluated a simple RandomForestClassifier.
5. Then using hyperparameter tuning with 5 fold cross-validation fitted 100 models to find the best random forest model for predicting disaster response category. Random Forest best parameters were
   <code>
   {'clf**criterion': 'entropy',
   'clf**max_depth': 40,
   'clf**max_features': 'auto',
   'clf**random_state': 42}
   </code>

6. Using this best model we've made train_classifier.py
