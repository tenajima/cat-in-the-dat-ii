import os
import pickle
from logging import getLogger
from typing import Union, List

import lightgbm as lgb
import numpy as np
import pandas as pd

from scripts.utils import delete_tmp_pickle

logger = getLogger(__name__)


def train_lgb(
    fold,
    params: dict,
    X: Union[pd.DataFrame, np.array],
    y: Union[pd.Series, np.array],
    seed: int,
    groups: Union[list, None] = None,
    load: bool = True,
):
    """lightgbmを用いてfoldごとのモデルを返す。

    Args:
        fold : sklearn.preprocessingのfold
        params (dict): lightgbmのparameter
        X (Union[pd.DataFrame, np.array]): 特徴行列
        y (Union[pd.Series, np.array]): 目的変数
        groups (Union[list, None], optional): groupKFoldのやつ. Defaults to None.
        load (bool, optional): tmpモデルをロードするかどうか. Defaults to True.

    Returns:
        List[lgb.Booster]: Boosterのリスト
    """
    if not load:
        delete_tmp_pickle()

    models: List[lgb.Booster] = []

    for fold_n, (trn_idx, val_idx) in enumerate(fold.split(X, y, groups=groups)):
        tmp_file_name = f"tmp_model_{fold_n + 1}_fold.pkl"

        if os.path.isfile(tmp_file_name):
            logger.info(f"found {tmp_file_name}! loading...")
            model = pickle.load(open(tmp_file_name, "rb"))
        else:
            logger.info(f"train {fold_n + 1}th model")
            trn_X = X.iloc[trn_idx]
            trn_y = y.iloc[trn_idx]
            trn_set = lgb.Dataset(trn_X, trn_y)

            val_X = X.iloc[val_idx]
            val_y = y.iloc[val_idx]
            val_set = lgb.Dataset(val_X, val_y)

            model = lgb.train(
                params,
                trn_set,
                valid_sets=[trn_set, val_set],
                num_boost_round=1000,
                early_stopping_rounds=100,
                verbose_eval=100,
            )
            pickle.dump(model, open(tmp_file_name, "wb"))

        models.append(model)

    delete_tmp_pickle()
    return models
