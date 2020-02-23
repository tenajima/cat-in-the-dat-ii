import gokart
import luigi
import pandas as pd
import lightgbm as lgb

from scripts.model.model_lightgbm import train_lgb
from scripts.train.preprocess import Preprocess, DataForML


class _TrainModelForNullImportance(gokart.TaskOnKart):
    model_name = luigi.Parameter()
    random_state = luigi.IntParameter()
    params = luigi.DictParameter()
    shuffle_y = luigi.BoolParameter()

    def requires(self):
        return Preprocess()

    def output(self):
        return self.make_target(f"./null_importance/{self.model_name}/importance.pkl")

    def run(self):
        data: DataForML = self.load()
        train_X = data.train_X
        train_y = data.train_y
        if self.shuffle_y:
            train_y = train_y.sample(frac=1)
        fold = data.fold

        models = train_lgb(
            fold, self.params.get_wrapped(), train_X, train_y, self.random_state
        )

        importance = pd.DataFrame()
        for i, model in enumerate(models):

            df_fold_importance = pd.DataFrame()
            df_fold_importance["feature"] = train_X.columns
            df_fold_importance["fold"] = i
            df_fold_importance["importance"] = model.feature_importance(
                importance_type="gain"
            )
            importance = pd.concat([importance, df_fold_importance])

        importance = importance.groupby("feature")[["importance"]].mean()
        importance = importance.sort_values("importance", ascending=False).reset_index()
        self.dump(importance)


class NullImportance(gokart.TaskOnKart):
    def requires(self):
        requires_ = dict()
        for i in range(100):
            model_name = "model_" + str(i).zfill(2)
            if i == 0:
                requires_[model_name] = _TrainModelForNullImportance(
                    model_name=model_name, shuffle_y=False
                )
            else:
                requires_[model_name] = _TrainModelForNullImportance(
                    model_name=model_name, shuffle_y=True
                )
        return requires_

    def output(self):
        return self.make_target("./train/models.pkl")

    def run(self):
        pass
