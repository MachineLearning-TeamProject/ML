import itertools

from preprocessing import *
from memory_based import *

from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
def user_total_visited_area(user_visit_rating_matrix):
    # 각 사용자별로 0이 아닌 지역 수를 계산
    non_zero_counts = user_visit_rating_matrix.astype(bool).sum(axis=1)

    # 결과를 데이터프레임으로 변환
    result_df = pd.DataFrame(non_zero_counts, columns=['total_visited_areas'])
    result_df.index.name = 'user_id'

    # 파일로 저장
    result_df.to_csv('dataset/evaluation/user_total_visited_area.csv')

    return result_df


def analyze_user_visits(user_total_visited_area):
    # 가장 방문지가 많은 유저의 방문 수 (n)
    n = user_total_visited_area['total_visited_areas'].max()
    # print(f"가장 방문지가 많은 유저의 방문 수 (n): {n}")

    # 1부터 n까지 각 방문지 수에 해당하는 유저 수 계산
    visit_counts = [user_total_visited_area[user_total_visited_area['total_visited_areas'] == i].shape[0] for i in range(1, n + 1)]

    return visit_counts


def count_users_per_area(user_visit_rating_matrix):
    # 각 방문지별로 방문한 유저 수 계산
    user_counts_per_area = user_visit_rating_matrix.astype(bool).sum(axis=0).tolist()

    return user_counts_per_area


def create_and_sort_visit_table(user_visit_rating_matrix):
    # 각 방문지별 방문한 유저 ID 배열 및 총 방문한 유저 수 계산
    visit_data = []
    for area_id in user_visit_rating_matrix.columns:
        visited_users = user_visit_rating_matrix.index[user_visit_rating_matrix[area_id].astype(bool)].tolist()
        total_visits = len(visited_users)
        visit_data.append((area_id, visited_users, total_visits))

    # 데이터프레임 생성
    visit_df = pd.DataFrame(visit_data, columns=['Area ID', 'Visited User IDs', 'Total Visits'])

    # 총 방문한 유저 수가 가장 큰 여행지 순으로 정렬
    visit_df_sorted = visit_df.sort_values(by='Total Visits', ascending=False)

    return visit_df_sorted

# loop: test_data에서 1명의 유저를 차례대로 가져온다. 그 유저를 target_user 이라고 한다. 모든 test_data 내의 user에 대해 반복한다.

# ============================================================================================
# 보충 필요
def user_based_evaluation_explanation(train_data, test_data):
    print("Start - User-Based Method Evaluating Method")

        # test_data에서 1명의 유저를 랜덤하게 가져오고 그 유저를 target_user이라고 한다.

        # target_user의 총 방문지 개수를 센다. 이를 target_user_total_visited_area 이라고 하자. 만약 target_user_total_visited_area이 1이라면, 아래 코드를 실행시키지 않고 다음 유저를 평가한다. 또한 n이 1인 유저는 따로 user_not_evaluating 배열에 저장한다.(방문지가 1곳인 유저는 추천성능을 테스트할 수 없기 때문이다)

        # target_user에 대한 방문지 평점 행렬을 test_data에서 가져오고 이를 target_user_rating_array 라고 한다.

        # target_user가 방문한 방문지 id 를 모두 배열에 저장한다. 이를 target_user_visited_area 라고 하자.

        # target_user가 방문한 방문지 n개 중 1개~n-1개를 택한다. 이때 택한 방문지 조합은 다른 조합과 겹치면 안된다. 이를 모두 target_user_visited_area_subset라고 한다.

        # target_user_similarity_array를 만든다. 처음 feature은 target_user_visited_area_subset이다. 2번째 feature에는 cosine_similarity가 온다. cosine_similarity 값은 일단 모두 value를 0으로 초기화한다.

        # loop: target_user_visited_area_subset에 있는 조합 하나를 꺼내서 subset 이라고 하자. 모든 subset에 대해 반복한다.

            # target_user_rating_array를 target_user_rating_array_subset에 깊은 복사한다.

            # target_user_rating_array_subset 에서, subset에 있는 방문지id 대한 rating값을 target_user_subset_actual에 저장한다. 첫번째 column에는 target_user의 id, 2번째 column은 방문지 id, 3번째 칼럼인 value는 실제 rating 값을 넣는다.

            # target_user_rating_array_subset 에서, subset에 있는 방문지id 대한 rating값을 0으로 바꾼다.

            # target_user_rating_array_subset 와 train_data 를 사용해 target_user와 train_data에 있는 유저들 간 피어슨 상관계수를 각각 모두 구하고 이를 target_user_pcc 라고 하자. (pcc = Pearson Correlation Coefficient)

            # target_user_pcc를 내림차순 정렬하여 가장 유사도가 높은 유저 k명을 뽑고 그 유저들의 pcc를 target_user_similar_user_pcc_array에, target_user_id, similar_user_id, pcc 를 feature로 가지도록 하여 저장한다.(k=10으로 한다)

            # 그 pcc를 이용해 target_user의 subset 방문지만에 대해 예측 평점을 구하고 target_user_subset_predict 라고 한다.

            # target_user_subset_actual 과 target_user_subset_predict 사이의 유사도를 Cosine_similarity를 사용해 구하고 이를 subset_cos_sim 이라고 한다.

            # target_user_similarity_array에서 subset에 해당하는 row를 찾고, value에 subset_cos_sim을 넣는다.

        # 루프가 종료되면 target_user_similarity_array 을 내림차순으로 정렬하고 리턴한다.  함수를 종료한다.


def user_based_evaluation(train_data, test_data):
    # 평가 테스트

    print("Start - User-Based Method Evaluating Method")

    user_not_evaluating = []

    while True:
        target_user = np.random.choice(test_data.index)
        # 대상 사용자의 총 방문지 수 계산
        target_user_total_visited_area = np.sum(test_data.loc[target_user] > 0)
        if target_user_total_visited_area <= 3:
            user_not_evaluating.append(target_user)
        else:
            break

    # 대상 사용자의 방문지 평점 행렬 추출
    target_user_rating_array = test_data.loc[target_user]
    target_user_visited_area = target_user_rating_array[target_user_rating_array > 0].index

    # target_user_similarity_array 초기화
    target_user_similarity_array = pd.DataFrame(index=pd.Series(dtype='object'), columns=['cosine_similarity'])

    # 타겟 유저가 방문한 곳만 체크
    target_user_rating_actual = target_user_rating_array[list(target_user_visited_area)]

    # 어차피 해당 유저가 방문하지 않은 곳은 userbased 추천 시스템에서 사용되지 않음
    # 그러니까 그냥 유저가 방문한 곳 만 ! 다른 유저랑 비교하면 되지 않나?
    # 왜? 어차피 나머지 다 0이잖아. 뭘 계산해도 차피 무시될텐데 굳이?
    # train_data 중 타겟 유저가 실제로 방문한 지역만 가져오자
    train_data_rating__where_target_user_visited = train_data[list(target_user_visited_area)]

    # 모든 조합에 대해 반복
    for subset_size in range(1, len(target_user_visited_area)):
        for subset in itertools.combinations(target_user_visited_area, subset_size):
            target_user_rating_array_subset = target_user_rating_actual.copy()

            # subset에 있는 VISIT_ID를 가져옴
            target_user_subset_actual = target_user_rating_array_subset[list(subset)]

            # subset 제외하고 나머지를 예측하는 성능을 측정하기 위해 0으로 변경해줌
            target_user_rating_array_subset[list(subset)] = 0

            # 피어슨 상관계수 계산
            target_user_pcc = train_data_rating__where_target_user_visited.apply(
                lambda x: pearsonr(x, target_user_rating_array_subset)[0], axis=1)

            # 피어슨 상관계수가 대부분 NaN이므로, NaN을 제거하자

            # 유사도가 높은 상위 k명의 유저 선택 X 상관계수 있는 사람이라면 전부 ㄱㄱ
            # k = 10
            target_user_similar_user_pcc_array = target_user_pcc.nlargest(k)

            # 예측 평점 계산
            target_user_predict = train_data_rating__where_target_user_visited.loc[
                target_user_similar_user_pcc_array.index].mul(
                target_user_similar_user_pcc_array.values, axis=0).mean(axis=0)

            # subset에 해당하는 예측값만 저장
            target_user_subset_predict = target_user_predict[list(subset)]

            # 여기서부턴 생각해야됨 이걸 cossim으로 할지 아님 RME 같은걸로 할지?
            # Cosine similarity 계산
            subset_cos_sim = \
                cosine_similarity([target_user_subset_actual.fillna(0)], [target_user_subset_predict.fillna(0)])[0][0]

            # 유사도 배열 업데이트
            target_user_similarity_array.loc[subset] = subset_cos_sim

    # 유사도 배열 정렬
    target_user_similarity_array = target_user_similarity_array.sort_values(by='cosine_similarity', ascending=False)
    return target_user_similarity_array


# ============================================================================================

from sklearn.metrics import mean_squared_error, mean_absolute_error
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def user_based_evaluation_one_user(table):
    # 평가 테스트
    print("Start - User-Based Method Evaluating Method")
    train_data, test_data = train_test_split(table, test_size=0.2)

    while True:
        target_user = np.random.choice(test_data.index)
        target_user_total_visited_area = np.sum(test_data.loc[target_user] > 0)
        if target_user_total_visited_area > 3:
            break

    user_ids = test_data.index
    target_user_actual_array = test_data.loc[target_user]

    # 이 부분에 대해 아래 user_based 코드에 맞게 데이터를 넣어야 함
    target_user_predict_array = user_based(table, target_user)
    # target_user_predict_array = user_based(table, user_ids)

    rmse_value = rmse(target_user_actual_array, target_user_predict_array)
    mae_value = mae(target_user_actual_array, target_user_predict_array)


    print(target_user_actual_array)
    print(target_user_predict_array)
    print(f"RMSE: {rmse_value}")
    print(f"MAE: {mae_value}")
    # RMSE: 0.6959378281031444
    # MAE: 0.08056303879136917



def user_based(train_data, user_id):
    print("Start User-Based Method")
    # 2307 rows x 2810 columns
    # Initialize a Nearest Neighbors model using cosine similarity for user-based collaborative filtering.
    knn = NearestNeighbors(metric='cosine', algorithm='brute')

    # Fit the model to the data in 'table', which contains user-item interactions.
    knn.fit(train_data.values)

    # Find the 21 nearest neighbors (including the user themselves) for each user in the dataset.
    distances, indices = knn.kneighbors(train_data.values, n_neighbors=21)

    # Convert distance metrics to cosine similarity scores.
    cosine_similarity_ = np.array(1 - distances)

    results = pd.DataFrame()

    # for user_id in user_ids:
    # Get the indices of 20 nearest users (excluding the user themselves) for the specified user_id.
    sim_user = indices.tolist()[train_data.index.tolist().index(user_id)][1:]

    result = train_data.T[user_id].copy()
    ## 가보지 않은 visit index 추출
    index = result[result == 0].index

    ## weighted ratings page 25
    ## cosine_sim * visit rating / sum(cosine_sim)
    # Calculate weighted ratings for recommendations.
    # Multiply the cosine similarity of the nearest users with their ratings and normalize.
    # This creates a weighted average based on similarity for each item.
    res = np.matmul(cosine_similarity_[train_data.index.tolist().index(user_id)][1:], train_data.iloc[sim_user]) \
          / np.where(np.matmul(cosine_similarity_[train_data.index.tolist().index(user_id)][1:], train_data.iloc[sim_user] != 0) < 0.5, 1,
                     np.matmul(cosine_similarity_[train_data.index.tolist().index(user_id)][1:], train_data.iloc[sim_user] != 0))

    # Convert the result into a DataFrame, filling missing values with 0.
    res = pd.DataFrame(res, index=train_data.columns, columns=[user_id]).loc[index].fillna(0)
    result.loc[index] = np.array(res).flatten()

    results = pd.concat([results, result], axis=1)

    # Save the resulting DataFrame to a CSV file for later use or analysis.
    # results.to_csv("dataset/evaluation/user_based.csv")

    return results


if __name__ == "__main__":
    # open the file
    visit_data = pd.read_csv("dataset/수도권/tn_visit_area_info_방문지정보_A.csv")
    travel_data = pd.read_csv("dataset/수도권/tn_travel_여행_A.csv")
    user_data = pd.read_csv("dataset/수도권/tn_traveller_master_여행객 Master_A.csv")

    # preprocessing
    processed_visit_data = process_table(visit_data, "visit")
    processed_travel_data = process_table(travel_data, "travel")
    processed_user_data = process_table(user_data, "user")
    table = merge_table(processed_visit_data, processed_travel_data, processed_user_data)

    # row : User, column : item
    user_visit_rating_matrix = table.pivot_table(index='TRAVELER_ID', columns='VISIT_ID', values='RATING').fillna(0)

    user_based_evaluation_one_user(user_visit_rating_matrix)

    # # 데이터셋 분석
    #
    # # 1. 유저가 방문한 방문지의 개수는 몇개일까?
    # user_total_visited_area = user_total_visited_area(user_visit_rating_matrix)
    # user_total_visited_area.to_csv("dataset/evaluation/user_total_visited_area.csv")
    # visit_count = analyze_user_visits(user_total_visited_area)
    # # print(visit_count)
    # # [720, 605, 433, 243, 140, 66, 39, 24, 10, 4, 7, 5, 2, 6, 2, 1]
    # # 즉 1군데만 방문한 사람이 무려 720명...
    # # 최대 16군데 방문한 사람이 1명
    # # 4군데 이상 방문한 사람이 그리 많지 않음
    #
    # # 2. 방문지별로 방문한 유저 수는 몇명일까?
    # count_users_per_area = count_users_per_area(user_visit_rating_matrix)
    # #count_users_per_area.to_csv("dataset/evaluation/count_users_per_area.csv")
    #
    # # 3. 가장 방문자가 많은 방문지 순으로, 방문한 유저 배열을 보고 싶다.
    # create_and_sort_visit_table = create_and_sort_visit_table(user_visit_rating_matrix)











