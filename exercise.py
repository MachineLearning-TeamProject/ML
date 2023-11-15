import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)

def drop_column(visit_data, travel_data, user_data):
    ## 날짜 합치기
    ## 일단 구성해놓긴 했는데 그냥 출발 달로 바꿔도 ㄱㅊ을 듯
    visit_data["VISIT_MM"] = np.where(visit_data["VISIT_START_YMD"].str.slice(start=5, stop=7) ==
                                         visit_data["VISIT_END_YMD"].str.slice(start=5, stop=7),
                                         visit_data["VISIT_START_YMD"].str.slice(start=5, stop=7),
                                         visit_data["VISIT_START_YMD"].str.slice(start=5, stop=7))
    ## visit_dataset
    # 'VISIT_AREA_ID', 'TRAVEL_ID', 'VISIT_AREA_NM', 'VISIT_START_YMD',
    # 'VISIT_END_YMD', 'LOTNO_ADDR', 'RESIDENCE_TIME_MIN',
    # 'VISIT_AREA_TYPE_CD', 'REVISIT_YN', 'DGSTFN', 'REVISIT_INTENTION',
    # 'RCMDTN_INTENTION'
    # TODO
    ## VISIT_ORDER는 필요없을 것 같아서 일단은 DROP
    visit_data = visit_data.drop(columns = ['VISIT_ORDER', 'ROAD_NM_ADDR', 'X_COORD', 'Y_COORD', 'ROAD_NM_CD','LOTNO_CD',
                                            'POI_ID','POI_NM', 'VISIT_CHC_REASON_CD', 'LODGING_TYPE_CD', 'SGG_CD', 'VISIT_START_YMD', 'VISIT_END_YMD',
                                            'VISIT_START_YMD', 'VISIT_END_YMD',
                                            'VISIT_AREA_ID'])
    ## Drop VISIT_AREA_TYPE_CD
    ##역/터미널/고속도로 휴게소(9), 상점(10), 식당/카페(11), 집(21), 친구/친지집(22), 사무실(23), 숙소(24)
    visit_data = visit_data[(visit_data.VISIT_AREA_TYPE_CD < 21) | (visit_data.VISIT_AREA_TYPE_CD > 24)]
    visit_data = visit_data[(visit_data.VISIT_AREA_TYPE_CD > 12) | (visit_data.VISIT_AREA_TYPE_CD < 9)]

    # 재방문여부 Y => 1, N => 0
    visit_data = visit_data.replace({'REVISIT_YN': 'Y'}, 1)
    visit_data = visit_data.replace({'REVISIT_YN': 'N'}, 0)

    ## noise 제거하기
    ## DROP null value 만족도, 재방문여부, 추천여부 없을 때
    visit_data = visit_data.dropna(subset=['REVISIT_YN', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION'])

    # ID Mapping
    # visit_data["VISIT_ID"] = 0
    visit_data_group = visit_data.groupby('VISIT_AREA_NM')
    for num, key in enumerate(visit_data_group.groups.keys()):
        visit_data.loc[visit_data_group.groups[key], 'VISIT_ID'] = num+1

    # Reordering
    visit_data = visit_data.reindex([visit_data.columns[-1]] + visit_data.columns[:-1].to_list(), axis=1)

    ## travel_dataset
    ## 위와 동일
    travel_data["VISIT_MM"] = np.where(travel_data["TRAVEL_START_YMD"].str.slice(start=5, stop=7) ==
                                      travel_data["TRAVEL_END_YMD"].str.slice(start=5, stop=7),
                                      travel_data["TRAVEL_START_YMD"].str.slice(start=5, stop=7),
                                      travel_data["TRAVEL_START_YMD"].str.slice(start=5, stop=7))

    # 'TRAVEL_ID', 'TRAVEL_NM', 'TRAVELER_ID', 'TRAVEL_START_YMD',
    # 'TRAVEL_END_YMD', 'MVMN_NM'
    travel_data = travel_data.drop(columns=['TRAVEL_PURPOSE', 'TRAVEL_PERSONA', 'TRAVEL_MISSION', 'TRAVEL_MISSION_CHECK',
                                            'TRAVEL_START_YMD','TRAVEL_END_YMD'])

    # 교통수단 MVMN_NM 자가용 => 1, 대중교통 => 2, nan => 0
    travel_data['MVMN_NM'] = travel_data['MVMN_NM'].fillna(0)
    travel_data = travel_data.replace({'MVMN_NM': '자가용'}, 1)
    travel_data = travel_data.replace({'MVMN_NM': '대중교통 등'}, 2)
    ## 자가용 대중교통 임베딩

    ## user_data
    # 'TRAVELER_ID', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5',
    # 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    # 'TRAVEL_STATUS_ACCOMPANY'
    user_data = user_data.drop(columns=['RESIDENCE_SGG_CD','GENDER','AGE_GRP','EDU_NM','EDU_FNSH_SE','MARR_STTS','FAMILY_MEMB','JOB_NM',
                                        'JOB_ETC','INCOME','HOUSE_INCOME','TRAVEL_TERM','TRAVEL_NUM','TRAVEL_LIKE_SIDO_1','TRAVEL_LIKE_SGG_1',
                                        'TRAVEL_LIKE_SIDO_2','TRAVEL_LIKE_SGG_2','TRAVEL_LIKE_SIDO_3','TRAVEL_LIKE_SGG_3','TRAVEL_STYL_2',
                                        'TRAVEL_STYL_4','TRAVEL_STATUS_RESIDENCE','TRAVEL_STATUS_DESTINATION','TRAVEL_STATUS_YMD','TRAVEL_MOTIVE_1'
                                        ,'TRAVEL_MOTIVE_2','TRAVEL_MOTIVE_3','TRAVEL_COMPANIONS_NUM'])


    print(visit_data.isnull().sum())
    print(travel_data.isnull().sum())
    print(user_data.isnull().sum())
    #
    # visit_data.to_csv("visit.csv")
    # travel_data.to_csv("travel.csv")
    # user_data.to_csv("user.csv")
    return visit_data, travel_data, user_data

# join 해야할까?
# -> 그냥 찾아 쓰면 되지않을까??
def join_dataset(visit_data, travel_data, user_data):
    print(visit_data.head(5))
    print(travel_data.head(5))
    print(user_data.head(5))

    return visit_data, travel_data, user_data

def main():
    visit_data = pd.read_csv("dataset/수도권/tn_visit_area_info_방문지정보_A.csv")
    travel_data = pd.read_csv("dataset/수도권/tn_travel_여행_A.csv")
    user_data = pd.read_csv("dataset/수도권/tn_traveller_master_여행객 Master_A.csv")
    # 안 쓸 것 같음
    # code_b_data = pd.read_csv(".dataset/수도권/tc_codeb_코드B.csv")
    visit_data, travel_data, user_data = drop_column(visit_data, travel_data, user_data)
    visit_data, travel_data, user_data = join_dataset(visit_data, travel_data, user_data)

if __name__ == "__main__":
    main()




