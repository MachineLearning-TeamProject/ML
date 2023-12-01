import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import warnings

warnings.filterwarnings(action = 'ignore')

THRESHOLD = 13

if __name__ == "__main__":
    
    # open the file 
    table = pd.read_csv("dataset/data_after_preprocessing/dataset.csv")

    # 미션 
    mission_name = ['쇼핑', '테마파크, 놀이시설, 동/식물원 방문', '역사 유적지 방문', '시티투어', '야외 스포츠, 레포츠 활동',
     '지역 문화예술/공연/전시시설 관람', '유흥/오락(나이트라이프)', '캠핑', '지역 축제/이벤트 참가', '온천/스파',
     '교육/체험 프로그램 참가', '드라마 촬영지 방문', '종교/성지 순례', 'Well-ness 여행', 'SNS 인생샷 여행',
     '호캉스 여행', '신규 여행지 발굴', '반려동물 동반 여행', '인플루언서 따라하기 여행', '친환경 여행(플로깅 여행)']
    
    mission_code = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 21, 22, 23, 24, 25, 26, 27]

    missions = {code: name for code, name in zip(mission_code, mission_name)}
    
    # RATING이 13 이상인 행을 선택
    selected_rows = table[table['RATING'] >= 13]

    # 선택된 행들을 순회하면서 missions 열을 활성화
    for index, row in selected_rows.iterrows():
        
        purpose_list = list(map(int, row['TRAVEL_PURPOSE'].strip(';').split(';')))
        for purpose in purpose_list:
            if purpose > 27 or (purpose > 13 and purpose < 21):
                continue
            # mission_column = f"missions{purpose}"
            mission_column = missions[purpose]
            table.loc[index, mission_column] = 1

    # NaN 값을 0으로 채움
    table = table.fillna(0)

    print(table)

    table.to_csv("dataset/data_after_preprocessing/content_based.csv")

    
    