import inspect
from typing import Dict, List

import gokart
import luigi
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from scripts.utils import reduce_mem_usage

from .dataset import GetDataSet


class FeatureFactory:
    def get_feature_instance(self, feature_name: str) -> gokart.TaskOnKart:

        """特徴量名を指定するとその特徴量クラスのインスタンスを返す
        Args:
            feature_name (str): 特徴量クラスの名前
        Returns:
            gokart.TaskOnKart
        """

        if feature_name in globals():
            return globals()[feature_name]()
        else:
            raise ValueError(f"{feature_name}って特徴量名は定義されてないよ!!!")

    def get_feature_task(self, features: List[str]) -> Dict[str, gokart.TaskOnKart]:
        tasks = {}
        for feature in features:
            tasks[feature] = self.get_feature_instance(feature)

        return tasks


class GetFeature(gokart.TaskOnKart):
    """ 特徴作成のための基底クラス """

    features = luigi.ListParameter()

    def feature_list(self) -> List[str]:
        """特徴量名リストを取得する"""
        lst: List[str] = []
        for name in globals():
            obj = globals()[name]
            if inspect.isclass(obj) and obj not in [
                LabelEncoder,
                tqdm,
                GetDataSet,
                FeatureFactory,
                GetFeature,
                Feature,
                OriginFeature,
                BinaryCategorical,
                Ordinary,
                OneHotEncode,
            ]:
                lst.append(obj.__name__)
        return lst

    def requires(self):
        ff = FeatureFactory()
        # もしparameterのfeaturesが空なら全部の特徴量を作る
        if not self.features:
            self.features = self.feature_list()
        return ff.get_feature_task(self.features)

    def output(self):
        # return self.make_target("./feature/feature.pkl")
        return self.make_large_data_frame_target("./feature/feature.zip")

    def run(self):
        data: pd.DataFrame = self.load("Target")

        for key in self.input().keys():
            if key == "Target":
                continue
            feature: pd.DataFrame = self.load(key)
            data = data.join(feature)

        self.dump(data)


# =================================================================================


class Feature(gokart.TaskOnKart):
    """ 基底クラス """

    index_columns = "id"
    predict_column = "target"

    def requires(self):
        return GetDataSet()


class OriginFeature(Feature):
    """ データセットのデータをそのまま使える特徴量。 """

    target_column = ""

    def run(self):
        required_columns = {self.index_columns, self.target_column}
        dataset = self.load_data_frame(
            required_columns=required_columns, drop_columns=True
        )
        dataset = dataset.set_index(self.index_columns)
        dataset = reduce_mem_usage(dataset)
        self.dump(dataset)


class BinaryCategorical(Feature):
    target_column = ""

    def run(self):
        required_columns = {self.index_columns, self.target_column}
        dataset = self.load_data_frame(
            required_columns=required_columns, drop_columns=True
        )
        dataset = dataset.set_index(self.index_columns)
        dataset[self.target_column] = (
            dataset[self.target_column].astype(str).fillna("nan")
        )

        encoder = LabelEncoder()
        dataset[self.target_column] = encoder.fit_transform(dataset[self.target_column])

        dataset = dataset.rename(
            columns={self.target_column: "BinaryCategorical_" + self.target_column}
        )

        dataset = reduce_mem_usage(dataset)
        self.dump(dataset)


class Ordinary(Feature):
    target_column: str = ""
    ordinary_map: dict = {}

    def run(self):
        required_columns = {self.index_columns, self.target_column}
        dataset = self.load_data_frame(
            required_columns=required_columns, drop_columns=True
        )
        dataset = dataset.set_index(self.index_columns)
        dataset[self.target_column] = dataset[self.target_column].map(self.ordinary_map)
        dataset = dataset.fillna(0)
        dataset[self.target_column] = dataset[self.target_column].astype(int)
        dataset = dataset.rename(
            columns={self.target_column: "Ordinary_" + self.target_column}
        )
        dataset = reduce_mem_usage(dataset)
        self.dump(dataset)


class OneHotEncode(Feature):
    target_column: str = ""

    def run(self):
        required_columns = {self.index_columns, self.target_column, "target"}
        dataset = self.load_data_frame(
            required_columns=required_columns, drop_columns=True
        )
        dataset = dataset.set_index(self.index_columns)

        train = dataset.loc[dataset[self.predict_column].notna(), self.target_column]
        test = dataset.loc[dataset[self.predict_column].isna(), self.target_column]

        categories = train.dropna().unique()

        train_dummied = pd.get_dummies(
            pd.Categorical(train, categories),
            prefix="OHE_" + self.target_column,
            dummy_na=True,
        )
        train_dummied.index = train.index

        test_dummied = pd.get_dummies(
            pd.Categorical(test, categories),
            prefix="OHE_" + self.target_column,
            dummy_na=True,
        )
        test_dummied.index = test.index
        result = reduce_mem_usage(pd.concat([train_dummied, test_dummied]).sort_index())
        self.dump(result)


# ===================================================================================


class Target(OriginFeature):
    target_column = "target"


class Bin0(BinaryCategorical):
    target_column = "bin_0"


class Bin1(BinaryCategorical):
    target_column = "bin_1"


class Bin2(BinaryCategorical):
    target_column = "bin_2"


class Bin3(BinaryCategorical):
    target_column = "bin_3"


class Bin4(BinaryCategorical):
    target_column = "bin_4"


class Ord0(Feature):
    def run(self):
        required_columns = {self.index_columns, "ord_0"}
        dataset = self.load_data_frame(
            required_columns=required_columns, drop_columns=True
        )
        dataset = dataset.set_index(self.index_columns)
        dataset = dataset.fillna(0)
        dataset["ord_0"] = dataset["ord_0"].astype(int)
        dataset = dataset.rename(columns={"ord_0": "Ordinary_ord_0"})

        dataset = reduce_mem_usage(dataset)
        self.dump(dataset)


class Ord1(Ordinary):
    target_column = "ord_1"
    ordinary_map = {
        "Novice": 1,
        "Contributor": 2,
        "Expert": 4,
        "Master": 5,
        "Grandmaster": 6,
    }


class Ord2(Ordinary):
    target_column = "ord_2"
    ordinary_map = {
        "Freezing": 1,
        "Cold": 2,
        "Warm": 3,
        "Hot": 4,
        "Boiling Hot": 5,
        "Lava Hot": 6,
    }


class Ord3(Ordinary):
    target_column = "ord_3"
    # aからoがkeyで1から15までのvalue
    ordinary_map = {chr(i): i - 96 for i in range(97, 112)}


class Ord4(Ordinary):
    target_column = "ord_4"
    # AからZまで1から26までの辞書
    ordinary_map = {chr(i): i - 64 for i in range(65, 91)}


class Ord5(Ordinary):
    def run(self):
        required_columns = {self.index_columns, "ord_5"}
        dataset = self.load_data_frame(
            required_columns=required_columns, drop_columns=True
        )
        dataset = dataset.set_index(self.index_columns)
        dataset["Ord5_1"] = dataset["ord_5"].map(
            lambda string: ord(string[0]), na_action="ignore"
        )
        dataset["Ord5_2"] = dataset["ord_5"].map(
            lambda string: ord(string[1]), na_action="ignore"
        )
        map_ord5 = {
            key: value
            for value, key in enumerate(sorted(dataset["ord_5"].dropna().unique()))
        }
        dataset["Ordinary_ord_5"] = dataset["ord_5"].map(map_ord5)
        dataset = dataset[["Ord5_1", "Ord5_2", "Ordinary_ord_5"]]
        dataset = dataset.fillna(0).astype(int)

        dataset = reduce_mem_usage(dataset)
        self.dump(dataset)


class DaySinCos(Feature):
    def run(self):
        required_columns = {self.index_columns, "day"}
        dataset = self.load_data_frame(
            required_columns=required_columns, drop_columns=True
        )
        dataset = dataset.set_index(self.index_columns)
        dataset["day_sin"] = np.sin(2 * np.pi * dataset["day"] / 7)
        dataset["day_cos"] = np.cos(2 * np.pi * dataset["day"] / 7)
        dataset = dataset[["day_sin", "day_cos"]].fillna(-10)
        dataset = reduce_mem_usage(dataset)
        self.dump(dataset)


class MonthSinCos(Feature):
    def run(self):
        required_columns = {self.index_columns, "month"}
        dataset = self.load_data_frame(
            required_columns=required_columns, drop_columns=True
        )
        dataset = dataset.set_index(self.index_columns)
        dataset["month_sin"] = np.sin(2 * np.pi * dataset["month"] / 12)
        dataset["month_cos"] = np.cos(2 * np.pi * dataset["month"] / 12)
        dataset = dataset[["month_sin", "month_cos"]].fillna(-10)
        dataset = reduce_mem_usage(dataset)
        self.dump(dataset)


class OHEBin0(OneHotEncode):
    target_column = "bin_0"


class OHEBin1(OneHotEncode):
    target_column = "bin_1"


class OHEBin2(OneHotEncode):
    target_column = "bin_2"


class OHEBin3(OneHotEncode):
    target_column = "bin_3"


class OHEBin4(OneHotEncode):
    target_column = "bin_4"


class Nom0(OneHotEncode):
    target_column = "nom_0"


class Nom1(OneHotEncode):
    target_column = "nom_1"


class Nom2(OneHotEncode):
    target_column = "nom_2"


class Nom3(OneHotEncode):
    target_column = "nom_3"


class Nom4(OneHotEncode):
    target_column = "nom_4"


# Nom5から9まではカテゴリ数が多すぎるので単純にOHEできない
