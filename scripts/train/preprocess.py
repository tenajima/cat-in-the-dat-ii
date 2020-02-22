import dataclasses
from typing import Union

import gokart
import luigi
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from scripts.feature.get_feature import GetFeature
from scripts.train.fold import GetFold


@dataclasses.dataclass
class DataForML:
    train_X: pd.DataFrame
    train_y: Union[list, pd.Series]
    test_X: pd.DataFrame
    fold: Union[KFold, StratifiedKFold, GroupKFold]
    groups: Union[pd.Series, list, None] = None


class Preprocess(gokart.TaskOnKart):
    use_columns = luigi.ListParameter()
    drop_columns = luigi.ListParameter()

    def requires(self):
        return {"fold": GetFold(), "feature": GetFeature()}

    def output(self):
        return self.make_target("./train/preprocessed_data.pkl")

    def run(self):
        fold = self.load("fold")

        if self.use_columns:
            required_columns = sorted(
                list({"id"} | set(self.use_columns) - set(self.drop_columns))
            )
            feature: pd.DataFrame = self.load_data_frame(
                "feature", required_columns=required_columns, drop_columns=True
            ).sort_index()
        else:
            feature: pd.DataFrame = self.load_data_frame("feature").sort_index()
            if self.drop_columns:
                feature = feature.drop(columns=self.drop_columns)
        train = feature[feature["target"].notna()].copy()
        test = feature[feature["target"].isna()].copy()

        X = train.drop(columns="target")
        y = train["target"]
        test_X = test.drop(columns="target")

        data = DataForML(X, y, test_X, fold)

        self.dump(data)
