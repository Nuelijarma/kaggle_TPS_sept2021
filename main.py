# import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# Debugging
from pdb import set_trace as bp
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

if __name__=="__main__":
    logger.debug("Loading training data.")
    data = pd.read_csv("data/train.csv")
    # train_id = train_data[["id"]] # Note that we don't need id.
    x = data.drop(["id", "claim"], axis=1)
    y = data[["claim"]]

    logger.debug("Processing training data.")
    logger.debug("- Handling NaN values.")
    x_med = x.median(axis=0)
    x.fillna(x_med, inplace=True)

    logger.debug("Train/test split.")
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    logger.debug("Training Gaussian Naive Bayes classifier.")
    clf = GaussianNB()
    clf.fit(train_x, train_y.values.ravel())

    logger.debug("Testing classifier.")
    print(f"test accuracy: {clf.score(test_x, test_y)}")

    logger.debug("Loading validation data.")
    val_data = pd.read_csv("data/test.csv")
    val_id = val_data[["id"]] # But here we do need id.
    val_x = val_data.drop(["id"], axis=1)

    logger.debug("Processing validation data.")
    logger.debug("- Handling NaN values.")
    val_x.fillna(x_med, inplace=True)

    logger.debug("Predicting test 'claim' value.")
    val_y = clf.predict(val_x)

    logger.debug("Writing output")
    val_y = pd.DataFrame(data=val_y.reshape((-1,1)), columns=["claim"])
    val_final = val_id.join([val_y])
    val_final.to_csv("naive_bayes_solution.csv", index=False)
