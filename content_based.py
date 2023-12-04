import pandas as pd
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def preprocessing_for_content_based():

    travel_table = pd.read_csv("dataset/data_after_preprocessing/dataset.csv")

    # 미션
    mission_name = ['쇼핑', '테마파크, 놀이시설, 동/식물원 방문', '역사 유적지 방문', '시티투어', '야외 스포츠, 레포츠 활동',
    '지역 문화예술/공연/전시시설 관람', '유흥/오락(나이트라이프)', '캠핑', '지역 축제/이벤트 참가', '온천/스파',
    '교육/체험 프로그램 참가', '드라마 촬영지 방문', '종교/성지 순례', 'Well-ness 여행', 'SNS 인생샷 여행',
    '호캉스 여행', '신규 여행지 발굴', '반려동물 동반 여행', '인플루언서 따라하기 여행', '친환경 여행(플로깅 여행)']

    mission_code = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27]

    missions = {code: name for code, name in zip(mission_code, mission_name)}

    # 방문지 유형 VISIT_AREA_TYPE_CD
    visit_area_type_name = ["자연관광지", "역사/유적/종교 시설 (문화재, 박물관, 촬영지, 절 등)", "문화시설(공연장, 영화관, 전시관 등)", "상업지구(거리, 시장, 쇼핑시설)", "레저/스포츠 관련 시설(스키, 카트, 수상레저)", "테마시설(놀이공원, 워터파크)", "산책로, 둘레길 등", "지역축제, 행사", "체험 활동 관광지"]
    
    visit_area_type_code = [1, 2, 3, 4, 5, 6, 7, 8, 13]

    visit_area_types = {code: name for code, name in zip(visit_area_type_code, visit_area_type_name)}
    
    # RATING이 13 이상인 행을 선택
    selected_rows = travel_table[travel_table['RATING'] >= 13]

    # 선택된 행들을 순회하면서 missions 열 및 visit area type 열을 활성화
    for index, row in selected_rows.iterrows():
        purpose_list = list(map(int, row['TRAVEL_PURPOSE'].strip(';').split(';')))
        for purpose in purpose_list:
            if purpose > 27 or (purpose > 13 and purpose < 21):
                continue
            mission_column = missions[purpose]
            travel_table.loc[index, mission_column] = 1
        
        visit_area_type = row['VISIT_AREA_TYPE_CD']
        visit_area_type_column = visit_area_types[visit_area_type]
        travel_table.loc[index, visit_area_type_column] = 1

    # NaN 값을 0으로 채움
    travel_table = travel_table.fillna(0)

    # user table 관련 tag
    user_table = pd.read_csv("dataset/data_after_preprocessing/dataset.csv")

    # 여행스타일에 따른 조건 함수 정의
    def apply_travel_style(row):
        rating_threshold = 14
        
        # 여행스타일1 (자연 도시)
        if (1 <= row['TRAVEL_STYL_1'] <= 2) and (row['RATING'] >= rating_threshold):
            row['자연'] = 1
        elif (6 <= row['TRAVEL_STYL_1'] <= 7) and (row['RATING'] >= rating_threshold):
            row['도시'] = 1
        
        # TRAVEL_STYL_3 (새로운 지역 익숙한 지역)
        elif (1 <= row['TRAVEL_STYL_3'] <= 2) and (row['RATING'] >= rating_threshold):
            row['새로운 지역'] = 1
        elif (6 <= row['TRAVEL_STYL_3'] <= 7) and (row['RATING'] >= rating_threshold):
            row['익숙한 지역'] = 1
        
        # TRAVEL_STYL_5 (휴양/휴식 체험활동)
        elif (1 <= row['TRAVEL_STYL_5'] <= 2) and (row['RATING'] >= rating_threshold):
            row['휴양/휴식'] = 1
        elif (6 <= row['TRAVEL_STYL_5'] <= 7) and (row['RATING'] >= rating_threshold):
            row['체험활동'] = 1
        
        # TRAVEL_STYL_6 (잘 알려지지 않은 방문지 알려진 방문지)
        elif (1 <= row['TRAVEL_STYL_6'] <= 2) and (row['RATING'] >= rating_threshold):
            row['잘 알려지지 않은 방문지'] = 1
        elif (6 <= row['TRAVEL_STYL_6'] <= 7) and (row['RATING'] >= rating_threshold):
            row['알려지지 않은 방문지'] = 1
        
        # TRAVEL_STYL_7 (계획에 따른 여행 상황에 따른 여행)
        elif (1 <= row['TRAVEL_STYL_7'] <= 2) and (row['RATING'] >= rating_threshold):
            row['계획에 따른 여행'] = 1
        elif (6 <= row['TRAVEL_STYL_7'] <= 7) and (row['RATING'] >= rating_threshold):
            row['상황에 따른 여행'] = 1
        
        # TRAVEL_STYL_8 (사진촬영 중요하지 않음 사진촬영 중요)
        elif (1 <= row['TRAVEL_STYL_8'] <= 2) and (row['RATING'] >= rating_threshold):
            row['사진촬영 중요하지 않음'] = 1
        elif (6 <= row['TRAVEL_STYL_8'] <= 7) and (row['RATING'] >= rating_threshold):
            row['사진촬영 중요'] = 1
        
        # 그 외의 경우
        return row

    # apply_travel_style 함수를 모든 행에 적용하여 각 열에 값을 할당
    user_table = user_table.apply(apply_travel_style, axis=1)

    # NaN 값을 0으로 채움
    user_table = user_table.fillna(0)

    # travel tag와 user tag를 포함하는 테이블 만들기
    merged_table = pd.merge(travel_table, user_table, on='VISIT_ID', how='inner')

    # # 행 정리
    travel_table_selected_feature = ['VISIT_ID', '쇼핑', '테마파크, 놀이시설, 동/식물원 방문', '역사 유적지 방문', '시티투어', '야외 스포츠, 레포츠 활동',
     '지역 문화예술/공연/전시시설 관람', '유흥/오락(나이트라이프)', '캠핑', '지역 축제/이벤트 참가', '온천/스파',
     '교육/체험 프로그램 참가', '드라마 촬영지 방문', '종교/성지 순례', 'Well-ness 여행', 'SNS 인생샷 여행',
     '호캉스 여행', '신규 여행지 발굴', '반려동물 동반 여행', '인플루언서 따라하기 여행', '친환경 여행(플로깅 여행)',
     "자연관광지", "역사/유적/종교 시설 (문화재, 박물관, 촬영지, 절 등)", "문화시설(공연장, 영화관, 전시관 등)", "상업지구(거리, 시장, 쇼핑시설)", "레저/스포츠 관련 시설(스키, 카트, 수상레저)",
       "테마시설(놀이공원, 워터파크)", "산책로, 둘레길 등", "지역축제, 행사", "체험 활동 관광지"]
    
    travel_table = travel_table[travel_table_selected_feature]

    user_table_selected_feature = ['VISIT_ID', '계획에 따른 여행', '도시', '사진촬영 중요', '사진촬영 중요하지 않음',
    '상황에 따른 여행', '새로운 지역', '알려지지 않은 방문지', '익숙한 지역',
    '자연', '잘 알려지지 않은 방문지', '체험활동', '휴양/휴식']
    
    user_table = user_table[user_table_selected_feature]
    
    merged_table_selected_feature = ['VISIT_ID', '쇼핑', '테마파크, 놀이시설, 동/식물원 방문', '역사 유적지 방문', '시티투어', '야외 스포츠, 레포츠 활동',
     '지역 문화예술/공연/전시시설 관람', '유흥/오락(나이트라이프)', '캠핑', '지역 축제/이벤트 참가', '온천/스파',
     '교육/체험 프로그램 참가', '드라마 촬영지 방문', '종교/성지 순례', 'Well-ness 여행', 'SNS 인생샷 여행',
     '호캉스 여행', '신규 여행지 발굴', '반려동물 동반 여행', '인플루언서 따라하기 여행', '친환경 여행(플로깅 여행)',
     '계획에 따른 여행', '도시', '사진촬영 중요', '사진촬영 중요하지 않음',
    '상황에 따른 여행', '새로운 지역', '알려지지 않은 방문지', '익숙한 지역',
    '자연', '잘 알려지지 않은 방문지', '체험활동', '휴양/휴식',
    "자연관광지", "역사/유적/종교 시설 (문화재, 박물관, 촬영지, 절 등)", "문화시설(공연장, 영화관, 전시관 등)", "상업지구(거리, 시장, 쇼핑시설)", "레저/스포츠 관련 시설(스키, 카트, 수상레저)", "테마시설(놀이공원, 워터파크)", "산책로, 둘레길 등", "지역축제, 행사", "체험 활동 관광지"]
    
    merged_table = merged_table[merged_table_selected_feature]

    # 같은 여행지인 경우, TAG값을 평균내서 적기
    travel_table = travel_table.groupby('VISIT_ID').agg('mean')
    user_table = user_table.groupby('VISIT_ID').agg('mean')
    merged_table = merged_table.groupby('VISIT_ID').agg('mean')
    

    # 평균값이 0.5 이상인 경우, TAG 행에 추가
    travel_table['TAG'] = travel_table.apply(lambda row: ', '.join(row.index[row >= 0.5]), axis=1)    
    user_table['TAG'] = user_table.apply(lambda row: ', '.join(row.index[row >= 0.5]), axis=1)    
    merged_table['TAG'] = merged_table.apply(lambda row: ', '.join(row.index[row >= 0.5]), axis=1)    

    # Resetting the index to include VISIT_ID as a regular column
    travel_table.reset_index(inplace=True)
    user_table.reset_index(inplace=True)
    merged_table.reset_index(inplace=True)

    travel_table = travel_table[['VISIT_ID', 'TAG']]
    user_table = user_table[['VISIT_ID', 'TAG']]
    merged_table = merged_table[['VISIT_ID', 'TAG']]

    # 결과를 CSV 파일로 저장
    travel_table.to_csv("dataset/data_after_preprocessing/content_based_only_travel.csv")
    user_table.to_csv("dataset/data_after_preprocessing/content_based_only_user.csv")
    merged_table.to_csv("dataset/data_after_preprocessing/content_based_combined.csv")

def recommend(table):
    
    # Drop rows where 'TAG' is NaN or 'np'
    table = table.dropna(subset=['TAG'])
    table = table[table['TAG'] != 'np']

    # Apply CountVectorizer to convert TAG into a matrix of token counts
    count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
    tag_matrix = count_vectorizer.fit_transform(table['TAG'])

    # Calculate cosine similarity between items (VISIT_IDs)
    cosine_sim = cosine_similarity(tag_matrix, tag_matrix)

    # Function to get recommendations based on the cosine similarity
    def get_recommendations(visit_id, cosine_sim=cosine_sim, table=table):
        # Get the index of the visit_id using .loc
        idx = table.loc[table['VISIT_ID'] == visit_id].index

        if not idx.empty:
            idx = idx[0]  # Access the first index if there are multiple matches

            # Get the pairwise similarity scores
            sim_scores = list(enumerate(cosine_sim[idx]))

            # Sort the visits based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get the top 5 most similar visits (excluding itself)
            top_similar_visits = [item[0] for item in sim_scores[1:11]]
            
            # Create a DataFrame with recommendations and corresponding cosine similarity
            recommendations = table.iloc[top_similar_visits, [0, -1]]  # Assuming the last column is 'TAG'
            
            similarity_scores = [item[1] for item in sim_scores[1:11]]
            recommendations['Cosine_Similarity'] = similarity_scores

            return recommendations
        else:
            print(f"VISIT_ID {visit_id} not found in the dataset.")
            return pd.DataFrame()

    # Example: Get recommendations for VISIT_ID 1
    print("Example: VISIT ID 1")
    print(table.iloc[0]["TAG"])
    recommendations = get_recommendations(1)
    print(recommendations)

    # Example: Get recommendations for TARGET_VISIT_ID
    TARGET_VISIT_ID = 3

    print("\nExample: VISIT ID ", TARGET_VISIT_ID)
    idx = table.loc[table['VISIT_ID'] == TARGET_VISIT_ID].index
    print(table.iloc[idx]["TAG"])
    recommendations = get_recommendations(TARGET_VISIT_ID)
    print(recommendations)

if __name__ == "__main__":

    # 전처리 필요시 아래 주석 해제
    # preprocessing_for_content_based()
    
    # Load the merged table from the CSV file
    travel_table = pd.read_csv("dataset/data_after_preprocessing/content_based_only_travel.csv")
    user_table = pd.read_csv("dataset/data_after_preprocessing/content_based_only_user.csv")
    merged_table = pd.read_csv("dataset/data_after_preprocessing/content_based_combined.csv")

    # Recommendation (Using only travel tag)
    print("-------------------------------------------")
    print("[1] Recommendation (Using only travel tag)")
    print("-------------------------------------------")

    recommend(travel_table)

    # Recommendation (Using only user tag)
    print("\n-------------------------------------------")
    print("[2] Recommendation (Using only user tag)")
    print("-------------------------------------------")
    
    recommend(user_table)

    # Recommendation (Using travel & user tag)
    print("\n-------------------------------------------")
    print("[3] Recommendation (Using travel & user tag)")
    print("-------------------------------------------")
    recommend(merged_table)
