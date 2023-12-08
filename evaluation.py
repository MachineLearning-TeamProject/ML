from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Root Mean Squared Error (RMSE) calculation function
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Mean Absolute Error (MAE) calculation function
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

# Evaluation function that calculates and prints RMSE, MAE, and Accuracy
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

# Recommendation function that returns a list of recommended items
def recommend(dataset, actual_table, predict_table, user_id, threshold):
    predict_table = predict_table.drop(actual_table.index[np.nonzero(actual_table * np.array(actual_table > 8.25))[0]])
    predict_table = predict_table.nlargest(10, user_id)
    recommend_index = list(predict_table[np.array(predict_table) > threshold].index)
    result = []
    for idx in recommend_index:
        result.append(np.array(dataset[dataset['VISIT_ID']==idx]['VISIT_AREA_NM'])[0])
    return result