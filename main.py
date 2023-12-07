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
        if value_index.shape[0] > 2:
            choice_num = value_index.shape[0] // 3
            val = np.random.choice(value_index, choice_num)
            test_mask[idx][val] = 0

    user_visit_rating_matrix_mask = test_mask * user_visit_rating_matrix
    return user_visit_rating_matrix_mask

def save_csv(area_code, **kwargs):
    for key, value in kwargs.items():
        value.to_csv(os.path.join("dataset","data_after_preprocessing", area[area_code], key)+".csv")


from sklearn.metrics import mean_squared_error, mean_absolute_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def evaluation_func(predict_table, actual_table, threshold):
    rmse_value = rmse(actual_table, predict_table)
    mae_value = mae(actual_table, predict_table)

    print(f"RMSE: {rmse_value}")
    print(f"MAE: {mae_value}")

    count_all = 0
    correct = 0
    for column in predict_table.columns:
        if list(np.nonzero(actual_table[column]))[0].shape[0] > 2:
            diff_list = np.isin(np.array(actual_table[column][actual_table[column] > 8.25].index),
                                np.array(predict_table[column][predict_table[column] > threshold].index))

            count_all += diff_list.shape[0]
            correct += np.count_nonzero(diff_list)

    print(f"Accuracy: {correct/count_all*100}%")
    print()

def recommend(dataset, actual_table, predict_table, user_id, threshold):
    # print(np.array(predict_table) > 8.25)
    predict_table = predict_table.drop(np.nonzero(actual_table * np.array(actual_table > 8.25))[0])
    predict_table = predict_table.nlargest(10, user_id)
    recommend_index = list(predict_table[np.array(predict_table) > threshold].index)
    for idx in recommend_index:
        print(np.array(dataset[dataset['VISIT_ID']==idx]['VISIT_AREA_NM'])[0])

if __name__ == "__main__":
    area_code = 1
    evaluation = False

    # open the file
    visit_data, travel_data, user_data = read_data(area_code)

    # preprocessing
    processed_visit_data = process_table(visit_data, "visit")
    processed_travel_data = process_table(travel_data, "travel")
    processed_user_data = process_table(user_data, "user")
    dataset = merge_table(processed_visit_data, processed_travel_data, processed_user_data)

    # row : User, column : item
    user_visit_rating_matrix = dataset.pivot_table(index='TRAVELER_ID', columns='VISIT_ID', values='RATING').fillna(0)

    if evaluation:
        rating_matrix = model_eval(user_visit_rating_matrix)
        rating_matrix_index = rating_matrix.index
    else:
        rating_matrix = user_visit_rating_matrix
        rating_matrix_index = ['a000012']

    # collaborative filtering
    user_based_result = user_based(rating_matrix.copy(), np.array(rating_matrix_index))
    evaluation_func(user_based_result.copy(), user_visit_rating_matrix.T[rating_matrix_index].copy(), 8.25)

    item_based_result = item_based(rating_matrix.T.copy(), np.array(rating_matrix_index))
    evaluation_func(item_based_result.copy(), user_visit_rating_matrix.T[rating_matrix_index].copy(), 8.25)

    # Model-based Filterting
    svd_result = singular_value_decomposition(rating_matrix.copy(), rating_matrix_index,n=1000)
    evaluation_func(svd_result.copy(), user_visit_rating_matrix.T[rating_matrix_index].copy(), 0.2)

    if evaluation:
        factorizer = MatrixFactorization(rating_matrix.copy(), k=3, learning_rate=0.01, reg_param=0.01, epochs=300, verbose=True)
        factorizer.fit()
        mf_result = factorizer.test(rating_matrix_index)
        evaluation_func(mf_result.copy(), user_visit_rating_matrix.T[rating_matrix_index].copy(), 8.25)
    else:
        factorizer = MatrixFactorization(rating_matrix.copy(), k=3, learning_rate=0.01, reg_param=0.01, epochs=50,verbose=True)
        factorizer.load_array()
        factorizer.fit()
        mf_result = factorizer.test(rating_matrix_index)

    if not evaluation:
        recommend(dataset, user_visit_rating_matrix.T[rating_matrix_index], user_based_result, rating_matrix_index, 8.25)
        recommend(dataset, user_visit_rating_matrix.T[rating_matrix_index], item_based_result, rating_matrix_index, 8.25)
        recommend(dataset, user_visit_rating_matrix.T[rating_matrix_index], svd_result, rating_matrix_index,0.2)
        recommend(dataset, user_visit_rating_matrix.T[rating_matrix_index], mf_result, rating_matrix_index, 8.25)


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
