import pandas as pd
import os
from preprocessing import process_table, merge_table
import numpy as np
from memory_based import user_based, item_based
from model_based import MatrixFactorization, singular_value_decomposition

area = {1: "수도권", 2: "동부권", 3: "서부권", 4: "도서산간"}

def read_data(key):
    visit_data = pd.read_csv(os.path.join("dataset", area[key], "tn_visit_area_info_방문지정보_A.csv"))
    travel_data = pd.read_csv(os.path.join("dataset", area[key], "tn_travel_여행_A.csv"))
    user_data = pd.read_csv(os.path.join("dataset", area[key], "tn_traveller_master_여행객 Master_A.csv"))

    return visit_data, travel_data, user_data

def model_eval(user_visit_rating_matrix):
    test_mask = np.ones(user_visit_rating_matrix.shape)
    for idx, i in enumerate(np.array(user_visit_rating_matrix)):
        value_index = list(np.nonzero(i))[0]
        if value_index.shape[0] > 3:
            choice_num = value_index.shape[0] // 3
            val = np.random.choice(value_index, choice_num)
            test_mask[idx][val] = 0

    user_visit_rating_matrix_mask = test_mask * user_visit_rating_matrix
    return user_visit_rating_matrix_mask

def save_csv(area_code, **kwargs):
    for key, value in kwargs.items():
        value.to_csv(os.path.join("dataset","data_after_preprocessing", area[area_code], key)+".csv")


if __name__ == "__main__":
    area_code = 1

    # open the file
    visit_data, travel_data, user_data = read_data(area_code)

    # preprocessing
    processed_visit_data = process_table(visit_data, "visit")
    processed_travel_data = process_table(travel_data, "travel")
    processed_user_data = process_table(user_data, "user")
    dataset = merge_table(processed_visit_data, processed_travel_data, processed_user_data)

    # row : User, column : item
    user_visit_rating_matrix = dataset.pivot_table(index='TRAVELER_ID', columns='VISIT_ID', values='RATING').fillna(0)

    evaluation = True
    if evaluation:
        rating_matrix = model_eval(user_visit_rating_matrix)
    else:
        rating_matrix = user_visit_rating_matrix

    # collaborative filtering
    user_based_result = user_based(rating_matrix.copy(), np.array(rating_matrix.index))
    item_based_result = item_based(rating_matrix.T.copy(), np.array(rating_matrix.index))

    # Model-based Filterting
    svd_result = singular_value_decomposition(rating_matrix.copy(), rating_matrix.index,n=1000)

    factorizer = MatrixFactorization(rating_matrix.copy(), k=3, learning_rate=0.01, reg_param=0.01, epochs=300, verbose=True)
    factorizer.fit()
    mf_result = factorizer.test(rating_matrix.index)


    # # save the file
    save_csv(area_code,
             dataset = dataset,
             processed_user_data = processed_user_data,
             processed_visit_data = processed_visit_data,
             processed_travel_data = processed_travel_data,
             rating_matrix = rating_matrix,
             user_based = user_based_result,
             item_based = item_based_result,
             svd = svd_result,
             MF = mf_result)
