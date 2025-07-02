import pandas as pd
import numpy as np
import logging
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

def main():
    train = pd.DataFrame.from_records(json.load(open('train.json'))) 
    test = pd.DataFrame.from_records(json.load(open('test.json')))
    train, validation = train_test_split(train, test_size=1/3, random_state=123)
    featurizer = ColumnTransformer(
        transformers=[ ("year", 'passthrough', ["year"]),
                       ("authors", TfidfVectorizer(analyzer='word', ngram_range=(1,1)), "authors"),
                      ],
        remainder='drop')
    
    dummy = make_pipeline(featurizer, DummyRegressor())
    ridge = make_pipeline(featurizer, Ridge(alpha=1))
    label = 'n_citation'
    for model_name, model in [("dummy", dummy),
                              ("ridge", ridge),
                              ]:
        
        logging.info(f"Fitting model {model_name}")
        model.fit(train.drop([label], axis=1), np.log1p(train[label].values))
        for split_name, split in [("train     ", train),
                                  ("validation", validation)]:
            pred = np.expm1(model.predict(split.drop([label], axis=1)))
            mae = mean_absolute_error(split[label], pred)
            logging.info(f"{model_name} {split_name} MAE: {mae:.2f}")
    predicted = np.expm1(ridge.predict(test))
    test['n_citation'] = predicted
    json.dump(test[['n_citation']].to_dict(orient='records'), open('predicted.json', 'w'), indent=2)
        
logging.getLogger().setLevel(logging.INFO)
main()
