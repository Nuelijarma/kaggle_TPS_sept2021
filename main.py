import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# Debugging
from pdb import set_trace as bp
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def main():
    logger.debug("Loading training data.")
    data = pd.read_csv("data/train.csv", index_col="id")
    x = data.drop(["claim"], axis=1)
    y = data[["claim"]]

    logger.debug("Processing training data.")
    logger.debug("- Adding NaN count as a feature.")
    x["na"] = x.isna().sum(axis=1)
    # logger.debug("- Handling NaN values.")
    # x_med = x.mean(axis=0)
    # x.fillna(x_med, inplace=True)
    logger.debug("- Normalizing features.")
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    logger.debug("Train/test split.")
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    logger.debug("Training XGB classifier.")
    clf = XGBClassifier()
    clf.fit(train_x, train_y.values.ravel())

    logger.debug("Testing classifier.")
    print(f"test score: {roc_auc_score(test_y, clf.predict_proba(test_x)[:,1])}")

    logger.debug("Loading validation data.")
    val_x = pd.read_csv("data/test.csv", index_col="id")
    val_index = val_x.index

    logger.debug("Processing validation data.")
    logger.debug("- Adding NaN count as a feature.")
    val_x["na"] = val_x.isna().sum(axis=1)
    # logger.debug("- Handling NaN values.")
    # val_x.fillna(x_med, inplace=True)
    logger.debug("- Normalizing features.")
    val_x = scaler.transform(val_x)

    logger.debug("Predicting test 'claim' value.")
    val_y = clf.predict_proba(val_x)

    logger.debug("Writing output")
    val_final = pd.DataFrame(data={"id": val_index.values, "claim": val_y[:,1]})
    val_final.to_csv("xgb_solution.csv", index=False)

if __name__=="__main__":
    main()
