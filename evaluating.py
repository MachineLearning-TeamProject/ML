import pandas as pd
from sklearn.model_selection import train_test_split

import preprocessing

from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

from memory_based import user_based


def predict_ratings(user_based_func, test_data, train_data):
    predictions = test_data.copy()
    for user in test_data.index:
        user_predictions = user_based_func(train_data, user)
        predictions.loc[user] = user_predictions
    return predictions


def evaluate(predictions, actual, mask):
    pred = predictions[mask == 1].fillna(0)
    act = actual[mask == 1].fillna(0)

    rmse = np.sqrt(mean_squared_error(act, pred))
    mae = mean_absolute_error(act, pred)

    return rmse, mae


if __name__ == "__main__":
    # open the file
    visit_data = pd.read_csv("dataset/수도권/tn_visit_area_info_방문지정보_A.csv")
    travel_data = pd.read_csv("dataset/수도권/tn_travel_여행_A.csv")
    user_data = pd.read_csv("dataset/수도권/tn_traveller_master_여행객 Master_A.csv")

    # preprocessing
    processed_visit_data = preprocessing.process_table(visit_data, "visit")
    processed_travel_data = preprocessing.process_table(travel_data, "travel")
    processed_user_data = preprocessing.process_table(user_data, "user")
    table = preprocessing.merge_table(processed_visit_data, processed_travel_data, processed_user_data)

    # row : User, column : item
    user_visit_rating_matrix = table.pivot_table(index='TRAVELER_ID', columns='VISIT_ID', values='RATING').fillna(0)

    # divide train/test set
    train_data, test_data = train_test_split(user_visit_rating_matrix, test_size=0.2)
    test_set_mask = test_data.copy()
    test_set_mask[test_set_mask != 0] = 1
    user_based_result = user_based(train_data.copy(), 'a007615')

    # Generate predictions
    predictions = predict_ratings(user_based, test_data, user_visit_rating_matrix)
    ## predictions = predict_ratings(user_based, test_data, train_data)

    # Evaluate
    rmse, mae = evaluate(predictions, test_data, test_set_mask)
    print(f'RMSE: {rmse}')
    print(f'MAE: {mae}')

## 실행 시 약 2분정도 걸립니다. 추후 더 수정할 예정...
## user a007267 에 대한 예측 정확도
## RMSE: 0.4410320909666542
## MAE: 0.01345666887481834