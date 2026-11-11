import pickle
from pathlib import Path

import pandas as pd
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


def roast_category(roast: object) -> int | float:
    if not isinstance(roast, str):
        return float("nan")
    return ROAST_CATEGORIES.get(roast, float("nan"))


def main() -> None:
    data = pd.read_csv(DATA_URL)
    y = data["rating"]

    lr = LinearRegression()
    lr.fit(data[["100g_USD"]], y)

    data["roast_cat"] = data["roast"].map(roast_category)
    dtr = DecisionTreeRegressor(random_state=0)
    dtr.fit(data[["100g_USD", "roast_cat"]].values, y)

    output_dir = Path(__file__).resolve().parent
    for model, filename in (
        (lr, "model_1.pickle"),
        (dtr, "model_2.pickle"),
    ):
        model_path = output_dir / filename
        with model_path.open("wb") as model_file:
            pickle.dump(model, model_file, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
