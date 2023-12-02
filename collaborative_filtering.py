from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np

def user_based(table, user_id):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(table.values)
    distances, indices = knn.kneighbors(table.values, n_neighbors=21)
    cosine_similarity_ = np.array(1 - distances)

    sim_user = indices.tolist()[table.index.tolist().index(user_id)][1:]

    ## weighted ratings page 25
    ## cosine_sim * visit rating / sum(cosine_sim)
    result = np.matmul(cosine_similarity_[table.index.tolist().index(user_id)][1:],table.iloc[sim_user])\
             /np.matmul(cosine_similarity_[table.index.tolist().index(user_id)][1:],table.iloc[sim_user]!=0)

    result = pd.DataFrame(result, index=table.columns, columns=[user_id]).fillna(0)

    result.to_csv("dataset/data_after_preprocessing/user_based.csv")
    ## Evaluation function
    ## 0이 아닌 값 : 기존에 있던 rating이랑 차이점 비교
    ## 0 : 기존에 0이였으므로 추천도가 높은거 "추천"

def item_based(table, user_id):
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(table.values)
    distances, indices = knn.kneighbors(table.values, n_neighbors=21)
    cosine_similarity_ = np.array(1 - distances)

    # 유사한 방문지 20개 뽑기 ##자기 제외
    sim_ind = np.array(indices)[:, 1:]

    ## weighted ratings page 49
    ## cosine_sim * visit rating / sum(cosine_sim)
    result = table[user_id]

    ## 가보지 않은 visit index 추출
    index = result[result == 0].index

    res = np.sum(np.array(np.sum(table, axis=1)/np.sum(table != 0, axis=1))[sim_ind]
                 *cosine_similarity_[:,1:], axis=1) /cosine_similarity_[:,1:].sum(axis=1)
    res = pd.DataFrame(res, index=result.index, columns=[user_id]).loc[index].fillna(0)
    result.loc[index] = np.array(res).flatten()

    result.to_csv("dataset/data_after_preprocessing/item_based.csv")

    ## Evaluation function
    ## 0이 아닌 값 : 기존에 있던 rating이랑 차이점 비교
    ## 0 : 기존에 0이였으므로 추천도가 높은거 "추천"