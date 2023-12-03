from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def user_based(table, user_id):

    # 2307 rows x 2810 columns
    # Initialize a Nearest Neighbors model using cosine similarity for user-based collaborative filtering.
    knn = NearestNeighbors(metric='cosine', algorithm='brute')

    # Fit the model to the data in 'table', which contains user-item interactions.
    knn.fit(table.values)

    # Find the 21 nearest neighbors (including the user themselves) for each user in the dataset.
    distances, indices = knn.kneighbors(table.values, n_neighbors=201)
    # Convert distance metrics to cosine similarity scores.
    cosine_similarity_ = np.array(1 - distances)

    # Get the indices of 20 nearest users (excluding the user themselves) for the specified user_id.
    sim_user = indices.tolist()[table.index.tolist().index(user_id)][1:]

    result = table.T[user_id].copy()
    ## 가보지 않은 visit index 추출
    index = result[result == 0].index
#2798

    ## weighted ratings page 25
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
    # Save the resulting DataFrame to a CSV file for later use or analysis.
    result.to_csv("dataset/data_after_preprocessing/user_based.csv")

    ## Evaluation function
    ## 0이 아닌 값 : 기존에 있던 rating이랑 차이점 비교
    ## 0 : 기존에 0이였으므로 추천도가 높은거 "추천"


def item_based(table, user_id):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(table.values)
    distances, indices = knn.kneighbors(table.values, n_neighbors=21)
    cosine_similarity_ = np.array(1 - distances)

    ## weighted ratings page 49
    ## cosine_sim * visit rating / sum(cosine_sim)
    result = table[user_id].copy()

    # 유사한 방문지 20개 뽑기 ##자기 제외
    sim_ind = np.array(indices)[:, 1:]

    ## 가보지 않은 visit index 추출
    index = result[result == 0].index

    res = np.sum(np.array(result)[np.array(sim_ind)]*cosine_similarity_[:,1:],axis=1) \
          / cosine_similarity_[:,1:].sum(axis=1)
    res = pd.DataFrame(res, index=result.index, columns=[user_id]).loc[index].fillna(0)
    result.loc[index] = np.array(res).flatten()

    result.to_csv("dataset/data_after_preprocessing/item_based.csv")

    ## Evaluation function
    ## 0이 아닌 값 : 기존에 있던 rating이랑 차이점 비교
    ## 0 : 기존에 0이였으므로 추천도가 높은거 "추천"