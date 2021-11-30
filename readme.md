# CS334 Course Project

### Installation
Navigate to the project directory and execute the following command:
```pip install requirements.txt```

### Project Directory

```
cs334-course-project/
-- legacy/                  Legacy code samples
-- php/                     Scripts used for web interface (see jerry.games/334)
-- clf_models/              Directory containing all models used
-- -- dt.py                 Decision tree model
-- -- lgr.py                Logistic Regression model
-- -- nb.py                 Naive Bayes model
-- -- rf.py                 Random forest model
-- hp_search/               Directory containing scripts for hyperparameter tuning
-- driver.py                Driver file to test all models
-- dt.txt                   Visualization of the decision tree model (see plot_dt.py)
-- feature_correlation.py   Graph generation script for feature correlation heatmap
-- feature_selection_notes.txt      Notes on methodology and reasoning behind feature selection
-- globals.py               Python file to hold global constants
-- hotel_booking.csv        Original hotel booking dataset
-- hotel_booking_processed.csv      Post-processed hotel booking dataset
-- plot_dt.py               Script to visualize the decision tree model
-- preprocessing.py         Train/test-split post-feature selection hotel booking dataset to 4 csv files
-- readme.md                This file
-- requirements.txt         Package list for setup
-- xTest.csv                Test features
-- xTrain.csv               Train features
-- yTest.csv                Test labels
-- yTrain.csv               Train labels
```