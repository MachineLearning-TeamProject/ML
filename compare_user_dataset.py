import pandas as pd

def drop_column(user_data):
    ## user_data
    # 'TRAVELER_ID', 'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5',
    # 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    # 'TRAVEL_STATUS_ACCOMPANY'
    user_data = user_data.drop(columns=['RESIDENCE_SGG_CD','GENDER','AGE_GRP','EDU_NM','EDU_FNSH_SE','MARR_STTS','FAMILY_MEMB','JOB_NM',
                                        'JOB_ETC','INCOME','HOUSE_INCOME','TRAVEL_TERM','TRAVEL_NUM','TRAVEL_LIKE_SIDO_1','TRAVEL_LIKE_SGG_1',
                                        'TRAVEL_LIKE_SIDO_2','TRAVEL_LIKE_SGG_2','TRAVEL_LIKE_SIDO_3','TRAVEL_LIKE_SGG_3','TRAVEL_STYL_2',
                                        'TRAVEL_STYL_4','TRAVEL_STATUS_RESIDENCE','TRAVEL_STATUS_DESTINATION','TRAVEL_STATUS_YMD','TRAVEL_MOTIVE_1'
                                        ,'TRAVEL_MOTIVE_2','TRAVEL_MOTIVE_3','TRAVEL_COMPANIONS_NUM'])
    #
    # visit_data.to_csv("visit.csv")
    # travel_data.to_csv("travel.csv")
    # user_data.to_csv("user.csv")
    return user_data

def main():
    # visit_data = pd.read_csv("dataset/수도권/tn_visit_area_info_방문지정보_A.csv")
    # travel_data = pd.read_csv("dataset/수도권/tn_travel_여행_A.csv")
    user_data_0 = pd.read_csv("dataset/수도권/tn_traveller_master_여행객 Master_A.csv")
    user_data_1 = pd.read_csv("dataset/동부권/tn_traveller_master_여행객 Master_B.csv")
    user_data_0 = drop_column(user_data_0)
    user_data_1 = drop_column(user_data_1)
    print(user_data_0)
    print(user_data_1)
    user_data_0.to_csv("user_0.csv")
    user_data_1.to_csv("user_1.csv")
if __name__ == "__main__":
    main()
