import pickle
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression


DATA_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/datasets/refs/heads/main/"
    "data/coffee_analysis.csv"
)

ROAST_CATEGORIES = {
    "Light": 0,
    "Medium-Light": 1,
    "Medium": 2,
    "Medium-Dark": 3,
    "Dark": 4,
}


def roast_category(roast):
    return ROAST_CATEGORIES.get(roast, np.nan)

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    # Load the data
    coffee_df = pd.read_csv(DATA_URL)

    # Train the Linear Regression model
    X = coffee_df[["100g_USD"]].values
    y = coffee_df["rating"].values
    lr = LinearRegression()
    lr.fit(X, y)

    #save model_1
    save_model(lr, "model_1.pickle")
    print("Linear Regression model saved as model_1.pickle")

    # Train the Decision Tree model
    coffee_df["roast_cat"] = coffee_df["roast"].map(roast_category)
    X_dtr = coffee_df[["100g_USD", "roast_cat"]].values
    dtr = DecisionTreeRegressor(random_state=0)
    dtr.fit(X_dtr, y)

    #save model_2
    save_model(dtr, "model_2.pickle")
    print("Decision Tree model saved as model_2.pickle")
