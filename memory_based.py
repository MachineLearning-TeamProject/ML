from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import time

def user_based(table, user_ids):
    print("Start User-Based Method")
    start_time = time.time()

    # Initialize a Nearest Neighbors model using cosine similarity for user-based collaborative filtering.
    knn = NearestNeighbors(metric='cosine', algorithm='brute')

    # Fit the model to the data in 'table', which contains user-item interactions.
    knn.fit(table.values)

    # Find the 101 nearest neighbors (including the user themselves) for each user in the dataset.
    distances, indices = knn.kneighbors(table.values, n_neighbors=101)

    # Convert distance metrics to cosine similarity scores.
    cosine_similarity_ = np.array(1 - distances)

    results = pd.DataFrame()
    for user_id in user_ids :
        # Get the indices of 100 nearest users (excluding the user themselves) for the specified user_id.
        sim_user = indices.tolist()[table.index.tolist().index(user_id)][1:]

        result = table.T[user_id].copy()

        # Extract non-rating VISIT ID
        index = result[result == 0].index

        ## cosine_sim * visit rating / sum(cosine_sim)
        # Calculate weighted ratings for recommendations.
        # Multiply the cosine similarity of the nearest users with their ratings and normalize.
        # This creates a weighted average based on similarity for each item.
        res = np.matmul(cosine_similarity_[table.index.tolist().index(user_id)][1:], table.iloc[sim_user]) \
              / np.where(np.matmul(cosine_similarity_[table.index.tolist().index(user_id)][1:], table.iloc[sim_user] != 0) < 0.5, 1,
                         np.matmul(cosine_similarity_[table.index.tolist().index(user_id)][1:], table.iloc[sim_user] != 0))

        # Convert the result into a DataFrame, filling missing values with 0.
        res = pd.DataFrame(res, index=table.columns, columns=[user_id]).loc[index].fillna(0)
        result.loc[index] = np.array(res).flatten()

        results = pd.concat([results, result], axis=1)

    print("End User-Based Method ", round(time.time() - start_time, 2), "sec")
    return results


def item_based(table, user_ids):
    print("Start Item-Based Method")
    start_time = time.time()

    # Initialize a Nearest Neighbors model using cosine similarity for item-based collaborative filtering.
    knn = NearestNeighbors(metric='cosine', algorithm='brute')

    # Fit the model to the data in 'table', which contains user-item interactions.
    knn.fit(table.values)

    # Find the 21 nearest neighbors (including the visit area themselves) for each visit area in the dataset.
    distances, indices = knn.kneighbors(table.values, n_neighbors=21)
    cosine_similarity_ = np.array(1 - distances)

    results = pd.DataFrame()
    for user_id in user_ids:
        ## cosine_sim * visit rating / sum(cosine_sim)
        result = table[user_id].copy()

        # 유사한 방문지 20개 뽑기 ##자기 제외
        sim_ind = np.array(indices)[:, 1:]

        # Extract non-rating VISIT ID
        index = result[result == 0].index

        # Calculate weighted ratings for recommendations.
        # Multiply the cosine similarity of the nearest visit area with their ratings and normalize.
        # This creates a weighted average based on similarity for each item.
        res = np.sum(np.array(result)[np.array(sim_ind)]*cosine_similarity_[:,1:],axis=1) \
              / cosine_similarity_[:,1:].sum(axis=1)
        res = pd.DataFrame(res, index=result.index, columns=[user_id]).loc[index].fillna(0)
        result.loc[index] = np.array(res).flatten()

        results = pd.concat([results, result], axis=1)

    print("End Item-Based Method ", round(time.time() - start_time, 2), "sec")
    return results