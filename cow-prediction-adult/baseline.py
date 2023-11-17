"""
Бейзлайн задачи прогнозирования удоев.
В качестве прогнозного значения каждого из 8 удоев подставляется медианное значение известных контрольных
удоев животного.
"""

import os
import json
from typing import Any

import tqdm
import numpy as np
import pandas as pd

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error






def fit() -> Any:
    """
    Обучить модель прогнозирования
    @return: Обученная модель прогнозирования
    """
    model = CatBoostRegressor()
    model.load_model("catboost.cbm")
    return model


def predict(model: Any, test_dataset_path: str) -> pd.DataFrame:
    """
    Построить прогноз с помощью модели прогнозирования для датасета, заданного по имени файла

    @param model: Обученная ранее модель прогнозирования
    @param test_dataset_path: Путь к тестовому датасету
    @return: Датафрейм с построенным прогнозом, заданного формата
    """

    #cat_cols = ['farm', 'farmgroup', 'birth_date', 'animal_id']
    #num_cols = ['lactation', 'calving_date', 'milk_yield_1', 'milk_yield_2', 'calving_month']

    target_cols = ['milk_yield_3', 'milk_yield_4', 'milk_yield_5', 'milk_yield_6', 'milk_yield_7', 'milk_yield_8', 'milk_yield_9', 'milk_yield_10']

    # feature_cols = cat_cols + num_cols

    test_dataset = pd.read_csv(test_dataset_path)


    test_dataset['calving_month'] = pd.to_datetime(test_dataset['calving_date']).dt.month
    test_dataset['calving_date'] = pd.to_datetime(test_dataset['calving_date'])

    #test_dataset = test_dataset[feature_cols]



    train = pd.read_csv("train.csv")
    pedigree_set = pd.read_csv("pedigree.csv")
    numeric_columns = train.select_dtypes(include=['float64']).columns

    train[numeric_columns] = train[numeric_columns].transform(lambda x: x.fillna(x.mean()))
    
    lactation_columns = ['milk_yield_1', 'milk_yield_2', 'milk_yield_3', 'milk_yield_4', 'milk_yield_5',
                     'milk_yield_6', 'milk_yield_7', 'milk_yield_8', 'milk_yield_9', 'milk_yield_10']

    mean_per_lactation = train.groupby('animal_id')[lactation_columns].mean().reset_index()
    pedigree_subset = pedigree_set[['animal_id', 'mother_id']]


    merged_data = pd.merge(test_dataset, pedigree_subset, on='animal_id', how='left')
    merged_data['is_mother_present'] = merged_data['mother_id'].isin(train['animal_id'])

    merged_data = pd.merge(merged_data, mean_per_lactation, left_on='mother_id', right_on = 'animal_id', how='left',  suffixes=('', '_mother'))
    for i in range(3, 11):
        merged_data[f'milk_yield_{i}_mother'] = merged_data[f'milk_yield_{i}']

    for i in range(1, 11):
        merged_data[f'milk_yield_{i}_mother'].fillna(merged_data.apply(lambda row: (row['milk_yield_1'] + row['milk_yield_2']) / 2, axis=1), inplace=True)

    cat_cols_mom = ['farm', 'farmgroup', 'birth_date', 'animal_id', 'is_mother_present']
    num_cols_mom = ['lactation', 'calving_date', 'milk_yield_1', 'milk_yield_2', 'calving_month', 'milk_yield_1_mother',
        'milk_yield_2_mother', 'milk_yield_3_mother', 'milk_yield_4_mother',
        'milk_yield_5_mother', 'milk_yield_6_mother', 'milk_yield_7_mother',
        'milk_yield_8_mother', 'milk_yield_9_mother', 'milk_yield_10_mother']

    feature_cols = cat_cols_mom + num_cols_mom



    final_preds = model.predict(merged_data[feature_cols])
    final_preds = pd.DataFrame(final_preds)
    final_preds.columns = target_cols

    prediction_df = pd.DataFrame({
        'animal_id': test_dataset['animal_id'],
        'lactation': test_dataset['lactation'],
    })

    
    for i in range(1, 3):
        prediction_df[f'milk_yield_{i}'] = test_dataset[f'milk_yield_{i}']
    for i in range(3, 11):
        prediction_df[f'milk_yield_{i}'] = final_preds[f'milk_yield_{i}']


    return prediction_df


if __name__ == '__main__':
    _model = fit()

    _submission = predict(_model, os.path.join('data', 'X_test_public.csv'))
    _submission.to_csv(os.path.join('data', 'submission.csv'), sep=',', index=False)

    # _submission_private = predict(_model, os.path.join('private', 'X_test_private.csv'))
    # _submission_private.to_csv(os.path.join('data', 'submission_private.csv'), sep=',', index=False)
