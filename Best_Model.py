import pandas as pd
import numpy as np
import logging
import json
from sklearn.model_selection import train_test_split, cross_val_score
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

    # Models with adjusted parameters
    random_forest = make_pipeline(
        featurizer, 
        RandomForestRegressor(
            random_state=123, 
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            n_estimators=100
        )
    )
    gradient_boosting = make_pipeline(
        featurizer, 
        GradientBoostingRegressor(
            random_state=123,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8
        )
    )

    # Label
    label = 'n_citation'

    # Training and evaluation
    for model_name, model in [("random_forest", random_forest),
                              ("gradient_boosting", gradient_boosting)]:
        
        logging.info(f"Fitting model {model_name}")
        model.fit(train.drop([label], axis=1), np.log1p(train[label].values))
        for split_name, split in [("train     ", train),
                                  ("validation", validation)]:
            pred = np.expm1(model.predict(split.drop([label], axis=1)))
            mae = mean_absolute_error(split[label], pred)
            logging.info(f"{model_name} {split_name} MAE: {mae:.2f}")

logging.getLogger().setLevel(logging.INFO)
main()
