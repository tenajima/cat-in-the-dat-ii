import dataclasses
from typing import Union
import gokart
import luigi
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold

from scripts.feature.get_feature import GetFeature


@dataclasses.dataclass
class DataForML:
    train_X: pd.DataFrame
    train_y: Union[list, pd.Series]
    test_X: pd.DataFrame
    fold: Union[KFold, StratifiedKFold, GroupKFold]
    groups: Union[pd.Series, list, None] = None


class Preprocess(gokart.TaskOnKart):
    n_split = luigi.IntParameter()
    random_state = luigi.IntParameter()
    use_columns = luigi.ListParameter()
    drop_columns = luigi.ListParameter()

    def requires(self):
        return GetFeature()

    def output(self):
        return self.make_target("./train/preprocessed_data.pkl")

    def run(self):
        if self.use_columns:
            required_columns = sorted(
                list(set(self.use_columns) - set(self.drop_columns))
            )
            feature: pd.DataFrame = self.load_data_frame(
                required_columns=required_columns, drop_columns=True
            )
        else:
            feature: pd.DataFrame = self.load_data_frame()
            if self.drop_columns:
                feature = feature.drop(columns=self.drop_columns)
        train = feature[feature["target"].notna()].copy()
        test = feature[feature["target"].isna()].copy()

        X = train.drop(columns="target")
        y = train["target"]
        test_X = test.drop(columns="target")

        fold = StratifiedKFold(
            n_splits=self.n_split, shuffle=True, random_state=self.random_state
        )
        data = DataForML(X, y, test_X, fold)

        self.dump(data)
