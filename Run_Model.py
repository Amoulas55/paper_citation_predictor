import pandas as pd
import numpy as np
import logging
import json
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler, FunctionTransformer

def main():
    # Load train data
    train = pd.DataFrame.from_records(json.load(open('train.json')))
    
    # Sample 1% of the data
    train = train.sample(frac=0.01, random_state=123)
    
    # Split sampled data into train and validation sets
    train, validation = train_test_split(train, test_size=1/3, random_state=123)

    # Feature transformation
    featurizer = ColumnTransformer(
        transformers=[
            ("year", 'passthrough', ["year"]),
            ("authors", TfidfVectorizer(analyzer='word', ngram_range=(1, 2)), "authors"),
            ("abstract_length", FunctionTransformer(lambda x: np.array([len(str(a).split()) for a in x]).reshape(-1, 1)), "abstract"),
            ("num_references", FunctionTransformer(lambda x: np.array([len(a) if isinstance(a, list) else 0 for a in x]).reshape(-1, 1)), "references"),
            ("title_length", FunctionTransformer(lambda x: np.array([len(str(a).split()) for a in x]).reshape(-1, 1)), "title"),
        ],
        remainder='drop'
    )

    # Models with hyperparameter tuning
    rf_param_grid = {
        'randomforestregressor__n_estimators': [50, 100, 200],
        'randomforestregressor__max_depth': [5, 10, 15],
        'randomforestregressor__min_samples_split': [4, 8, 12],
        'randomforestregressor__min_samples_leaf': [2, 4, 6]
    }
    rf_pipeline = make_pipeline(featurizer, RandomForestRegressor(random_state=123))
    rf_grid_search = GridSearchCV(rf_pipeline, param_grid=rf_param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)

    gb_param_grid = {
        'gradientboostingregressor__n_estimators': [100, 200, 300],
        'gradientboostingregressor__learning_rate': [0.01, 0.05, 0.1],
        'gradientboostingregressor__max_depth': [3, 4, 5],
        'gradientboostingregressor__min_samples_split': [3, 5, 7],
        'gradientboostingregressor__min_samples_leaf': [2, 3, 4],
        'gradientboostingregressor__subsample': [0.7, 0.8, 0.9]
    }
    gb_pipeline = make_pipeline(featurizer, GradientBoostingRegressor(random_state=123))
    gb_grid_search = GridSearchCV(gb_pipeline, param_grid=gb_param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)

    # Label
    label = 'n_citation'

    # Training and evaluation
    for model_name, model_grid_search in [("random_forest", rf_grid_search),
                                          ("gradient_boosting", gb_grid_search)]:
        
        logging.info(f"Fitting model {model_name} with hyperparameter tuning")
        model_grid_search.fit(train.drop([label], axis=1), np.log1p(train[label].values))
        best_model = model_grid_search.best_estimator_
        logging.info(f"Best parameters for {model_name}: {model_grid_search.best_params_}")
        
        for split_name, split in [("train     ", train),
                                  ("validation", validation)]:
            pred = np.expm1(best_model.predict(split.drop([label], axis=1)))
            mae = mean_absolute_error(split[label], pred)
            logging.info(f"{model_name} {split_name} MAE: {mae:.2f}")

logging.getLogger().setLevel(logging.INFO)
main()
