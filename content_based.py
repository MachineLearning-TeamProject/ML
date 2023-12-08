import pandas as pd
import pandas as pd


def preprocessing_for_content_based(region):
    """
    Preprocesses the data for content-based recommendation.

    Reads the travel data from a CSV file, performs data manipulation and feature engineering,
    and saves the preprocessed data to separate CSV files.

    Args:
    - region (str): The region for which the data is being preprocessed.

    Returns:
    None
    """

    # Read travel data from the specified CSV file
    travel_table = pd.read_csv(f"dataset/data_after_preprocessing/{region}/dataset.csv")

    # Define mission names and codes
    mission_name = ['쇼핑', '테마파크, 놀이시설, 동/식물원 방문', '역사 유적지 방문', '시티투어', '야외 스포츠, 레포츠 활동',
    '지역 문화예술/공연/전시시설 관람', '유흥/오락(나이트라이프)', '캠핑', '지역 축제/이벤트 참가', '온천/스파',
    '교육/체험 프로그램 참가', '드라마 촬영지 방문', '종교/성지 순례', 'Well-ness 여행', 'SNS 인생샷 여행',
    '호캉스 여행', '신규 여행지 발굴', '반려동물 동반 여행', '인플루언서 따라하기 여행', '친환경 여행(플로깅 여행)']

    mission_code = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27]

    missions = {code: name for code, name in zip(mission_code, mission_name)}

    # Define visit area type names and codes
    visit_area_type_name = ["자연관광지", "역사/유적/종교 시설 (문화재, 박물관, 촬영지, 절 등)", "문화시설(공연장, 영화관, 전시관 등)", "상업지구(거리, 시장, 쇼핑시설)", "레저/스포츠 관련 시설(스키, 카트, 수상레저)", "테마시설(놀이공원, 워터파크)", "산책로, 둘레길 등", "지역축제, 행사", "체험 활동 관광지"]
    
    visit_area_type_code = [1, 2, 3, 4, 5, 6, 7, 8, 13]

    visit_area_types = {code: name for code, name in zip(visit_area_type_code, visit_area_type_name)}
    
    # Select rows where RATING is 13 or higher

    selected_rows = travel_table[travel_table['RATING'] >= 13]

    # Iterate through selected rows and activate mission and visit area type columns
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

    # Fill NaN values with 0
    travel_table = travel_table.fillna(0)

    # Read user table data from the specified CSV file
    user_table = pd.read_csv(f"dataset/data_after_preprocessing/{region}/dataset.csv")

    # Define a function for applying travel style conditions
    def apply_travel_style(row):
        rating_threshold = 14
        
        # Travel style 1 (Nature, City)
        if (1 <= row['TRAVEL_STYL_1'] <= 2) and (row['RATING'] >= rating_threshold):
            row['자연'] = 1
        elif (6 <= row['TRAVEL_STYL_1'] <= 7) and (row['RATING'] >= rating_threshold):
            row['도시'] = 1
        
        # Travel style 3 (New area, Familiar area)
        elif (1 <= row['TRAVEL_STYL_3'] <= 2) and (row['RATING'] >= rating_threshold):
            row['새로운 지역'] = 1
        elif (6 <= row['TRAVEL_STYL_3'] <= 7) and (row['RATING'] >= rating_threshold):
            row['익숙한 지역'] = 1
        
        # Travel style 5 (Leisure/Rest, Experience activities)
        elif (1 <= row['TRAVEL_STYL_5'] <= 2) and (row['RATING'] >= rating_threshold):
            row['휴양/휴식'] = 1
        elif (6 <= row['TRAVEL_STYL_5'] <= 7) and (row['RATING'] >= rating_threshold):
            row['체험활동'] = 1
        
        # Travel style 6 (Unknown visit place, Known visit place)
        elif (1 <= row['TRAVEL_STYL_6'] <= 2) and (row['RATING'] >= rating_threshold):
            row['잘 알려지지 않은 방문지'] = 1
        elif (6 <= row['TRAVEL_STYL_6'] <= 7) and (row['RATING'] >= rating_threshold):
            row['알려지지 않은 방문지'] = 1
        
        # Travel style 7 (Planned travel, Situation-based travel)
        elif (1 <= row['TRAVEL_STYL_7'] <= 2) and (row['RATING'] >= rating_threshold):
            row['계획에 따른 여행'] = 1
        elif (6 <= row['TRAVEL_STYL_7'] <= 7) and (row['RATING'] >= rating_threshold):
            row['상황에 따른 여행'] = 1
        
        # Travel style 8 (Not important in photography, Important in photography)
        elif (1 <= row['TRAVEL_STYL_8'] <= 2) and (row['RATING'] >= rating_threshold):
            row['사진촬영 중요하지 않음'] = 1
        elif (6 <= row['TRAVEL_STYL_8'] <= 7) and (row['RATING'] >= rating_threshold):
            row['사진촬영 중요'] = 1
        
        # For other cases
        return row

    # Apply the apply_travel_style function to all rows and assign values to each column
    user_table = user_table.apply(apply_travel_style, axis=1)

    # Fill NaN values with 0
    user_table = user_table.fillna(0)

    # Create a table containing travel and user tags
    merged_table = pd.merge(travel_table, user_table, on='VISIT_ID', how='inner')

    # Select specific features for each table
    travel_table_selected_feature = ['VISIT_ID', '쇼핑', '테마파크, 놀이시설, 동/식물원 방문', '역사 유적지 방문', '시티투어', '야외 스포츠, 레포츠 활동',
     '지역 문화예술/공연/전시시설 관람', '유흥/오락(나이트라이프)', '캠핑', '지역 축제/이벤트 참가', '온천/스파',
     '교육/체험 프로그램 참가', '드라마 촬영지 방문', '종교/성지 순례', 'Well-ness 여행', 'SNS 인생샷 여행',
     '호캉스 여행', '신규 여행지 발굴', '반려동물 동반 여행', '인플루언서 따라하기 여행', '친환경 여행(플로깅 여행)',
     "자연관광지", "역사/유적/종교 시설 (문화재, 박물관, 촬영지, 절 등)", "문화시설(공연장, 영화관, 전시관 등)", "상업지구(거리, 시장, 쇼핑시설)", "레저/스포츠 관련 시설(스키, 카트, 수상레저)",
       "테마시설(놀이공원, 워터파크)", "산책로, 둘레길 등", "지역축제, 행사", "체험 활동 관광지"]
    
    travel_table = travel_table[travel_table_selected_feature]

    user_table_selected_feature = ['VISIT_ID','계획에 따른 여행', '도시', '사진촬영 중요', '사진촬영 중요하지 않음',
    '상황에 따른 여행', '새로운 지역', '알려지지 않은 방문지', '익숙한 지역',
    '자연', '잘 알려지지 않은 방문지', '체험활동', '휴양/휴식']
    
    user_table = user_table[user_table_selected_feature]
    
    merged_table_selected_feature = ['VISIT_ID','쇼핑', '테마파크, 놀이시설, 동/식물원 방문', '역사 유적지 방문', '시티투어', '야외 스포츠, 레포츠 활동',
     '지역 문화예술/공연/전시시설 관람', '유흥/오락(나이트라이프)', '캠핑', '지역 축제/이벤트 참가', '온천/스파',
     '교육/체험 프로그램 참가', '드라마 촬영지 방문', '종교/성지 순례', 'Well-ness 여행', 'SNS 인생샷 여행',
     '호캉스 여행', '신규 여행지 발굴', '반려동물 동반 여행', '인플루언서 따라하기 여행', '친환경 여행(플로깅 여행)',
     '계획에 따른 여행', '도시', '사진촬영 중요', '사진촬영 중요하지 않음',
    '상황에 따른 여행', '새로운 지역', '알려지지 않은 방문지', '익숙한 지역',
    '자연', '잘 알려지지 않은 방문지', '체험활동', '휴양/휴식',
    "자연관광지", "역사/유적/종교 시설 (문화재, 박물관, 촬영지, 절 등)", "문화시설(공연장, 영화관, 전시관 등)", "상업지구(거리, 시장, 쇼핑시설)", "레저/스포츠 관련 시설(스키, 카트, 수상레저)", "테마시설(놀이공원, 워터파크)", "산책로, 둘레길 등", "지역축제, 행사", "체험 활동 관광지"]
    
    merged_table = merged_table[merged_table_selected_feature]

    # For the same travel destinations, average the TAG values
    travel_table = travel_table.groupby('VISIT_ID').agg('mean')
    user_table = user_table.groupby('VISIT_ID').agg('mean')
    merged_table = merged_table.groupby('VISIT_ID').agg('mean')

    # Add TAG rows for values with an average greater than or equal to 0.5
    travel_table['TAG'] = travel_table.apply(lambda row: ', '.join(row.index[row >= 0.5]), axis=1)
    user_table['TAG'] = user_table.apply(lambda row: ', '.join(row.index[row >= 0.5]), axis=1)
    merged_table['TAG'] = merged_table.apply(lambda row: ', '.join(row.index[row >= 0.5]), axis=1)

    # Reset the index to include VISIT_ID as a regular column
    travel_table.reset_index(inplace=True)
    user_table.reset_index(inplace=True)
    merged_table.reset_index(inplace=True)

    travel_table = travel_table[['VISIT_ID','TAG']]
    user_table = user_table[['VISIT_ID','TAG']]
    merged_table = merged_table[['VISIT_ID','TAG']]

    # Save results to CSV files
    travel_table.to_csv(f"dataset/data_after_preprocessing/{region}/content_based_only_travel.csv")
    user_table.to_csv(f"dataset/data_after_preprocessing/{region}/content_based_only_user.csv")
    merged_table.to_csv(f"dataset/data_after_preprocessing/{region}/content_based_combined.csv")

if __name__ == "__main__":
    # Uncomment the line below if preprocessing is necessary
    # preprocessing_for_content_based("수도권")