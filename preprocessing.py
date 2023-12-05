import pandas as pd
import numpy as np
import warnings

from sklearn.model_selection import train_test_split

from memory_based import user_based, item_based

from model_based import MatrixFactorization, singular_value_decomposition

warnings.filterwarnings(action = 'ignore')
# -----------------
# Utility Functions
# -----------------
def merge_dates(table, date_column, new_column_name):
    table[new_column_name] = table[date_column].str.slice(start=5, stop=7)
    return table

def drop_columns(table, columns_to_drop):
    return table.drop(columns=columns_to_drop)

def filter_by_column(table, column_name, conditions):
    return table[conditions]

def replace_values(table, column_name, value_mapping):
    return table.replace({column_name: value_mapping})

def drop_null_values(table, columns_to_drop_na):
    return table.dropna(subset=columns_to_drop_na)

def map_ids(table, group_by_column, id_column):
    table_group = table.groupby(group_by_column)
    for num, key in enumerate(table_group.groups.keys()):
        table.loc[table_group.groups[key], id_column] = num + 1

    # Reordering
    table = table.reindex([table.columns[-1]] + table.columns[:-1].to_list(), axis=1)
    return table

def drop_non_spots(table):
    num_of_rows_before = len(table)
    table_with_only_touristSpots = table[(table['VISIT_AREA_TYPE_CD'] < 9) | (table['VISIT_AREA_TYPE_CD'] == 13)]
    num_of_rows_after = len(table_with_only_touristSpots)
    print(f"방문지 유형 코드가 1~8 혹은 13이 아닌 방문지 {num_of_rows_before - num_of_rows_after}개를 삭제했습니다.")
    return table_with_only_touristSpots

def same_name_same_id(table):
    # table['VISIT_ID'] = table.groupby('VISIT_AREA_NM')['VISIT_AREA_ID'].transform('min')
    
    # VISIT_AREA_NM으로 groupby한 뒤, 앞에서부터 visit_id를 부여
    # print(table.groupby('VISIT_AREA_NM').cumcount() +1 .head())
    table['VISIT_ID'] = table.groupby('VISIT_AREA_NM').ngroup() + 1
    
    is_unique = not table['VISIT_ID'].duplicated().any()

    print(f"All values in 'VISIT_ID' are unique: {is_unique}")
    # Reordering
    # table = table.reindex([table.columns[-1]] + table.columns[:-1].to_list(), axis=1)
    return table
 
def make_user_feature(table, ids, feature1, feature2, feature3):
    table.loc[ids, "NUMBER_OF_PERSON"] = feature1
    table.loc[ids, "CHILDREN"] = feature2
    table.loc[ids, "PARENTS"] = feature3
    return table

# 나홀로 여행 -> 인원수:1 / 자녀 동반 여부: false / 부모 동반여부: false
def split_feature(table):
    for ids, data in enumerate(table['TRAVEL_STATUS_ACCOMPANY']):
        if not data.find("나홀로"):
            table = make_user_feature(table, ids, 1, 0, 0)
        elif not data.find("자녀"):
            table = make_user_feature(table, ids, 3, 1, 0)
        elif not data.find("2"):
            table = make_user_feature(table, ids, 2, 0, 0)
        elif not data.find("가족 외"):
            table = make_user_feature(table, ids, 3, 0, 1)
        else:
            table = make_user_feature(table, ids, 3, 0, 0)
    # table = drop_columns(table, "TRAVEL_STATUS_ACCOMPANY")

    return table

# -------------------------
# Main Table Processing
# -------------------------
def process_table(table, table_name):
    # 날짜 합치기
    if table_name == "visit":
        table = merge_dates(table, "VISIT_START_YMD", "VISIT_MM")
    elif table_name == "travel":
        table = merge_dates(table, "TRAVEL_START_YMD", "TRAVEL_MM")

    # 각 테이블에 따른 추가 제거할 열
    if table_name == 'visit':
        columns_to_drop = ['VISIT_ORDER', 'ROAD_NM_ADDR', 'X_COORD', 'Y_COORD', 'ROAD_NM_CD','LOTNO_CD',
                                            'POI_ID','POI_NM', 'VISIT_CHC_REASON_CD', 'LODGING_TYPE_CD', 'SGG_CD', 'VISIT_START_YMD', 'VISIT_END_YMD',
                                            'VISIT_START_YMD', 'VISIT_END_YMD']
                                            
    elif table_name == 'travel': 
        # columns_to_drop = ['TRAVEL_PURPOSE', 'TRAVEL_PERSONA', 'TRAVEL_MISSION', 'TRAVEL_MISSION_CHECK',
        #                                     'TRAVEL_START_YMD','TRAVEL_END_YMD']
        columns_to_drop = ['TRAVEL_PERSONA', 'TRAVEL_MISSION_CHECK',
                                            'TRAVEL_START_YMD','TRAVEL_END_YMD']
    elif table_name == 'user':
        columns_to_drop = ['RESIDENCE_SGG_CD','GENDER','AGE_GRP','EDU_NM','EDU_FNSH_SE','MARR_STTS','FAMILY_MEMB','JOB_NM',
                                        'JOB_ETC','INCOME','HOUSE_INCOME','TRAVEL_TERM','TRAVEL_NUM','TRAVEL_LIKE_SIDO_1','TRAVEL_LIKE_SGG_1',
                                        'TRAVEL_LIKE_SIDO_2','TRAVEL_LIKE_SGG_2','TRAVEL_LIKE_SIDO_3','TRAVEL_LIKE_SGG_3','TRAVEL_STYL_2',
                                        'TRAVEL_STYL_4','TRAVEL_STATUS_RESIDENCE','TRAVEL_STATUS_DESTINATION','TRAVEL_STATUS_YMD','TRAVEL_MOTIVE_1'
                                        ,'TRAVEL_MOTIVE_2','TRAVEL_MOTIVE_3','TRAVEL_COMPANIONS_NUM']
    # 특정 열 제거
    table = drop_columns(table, columns_to_drop)

    # VISIT_AREA_TYPE_CD 필터링
    if table_name == "visit":
        table = drop_non_spots(table)

    # REVISIT_YN 처리
    if 'REVISIT_YN' in table.columns:
        table = replace_values(table, 'REVISIT_YN', {'Y': 1, 'N': 0})

    # Drop null values
    if 'REVISIT_YN' in table.columns:
        table = drop_null_values(table, ['REVISIT_YN', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION'])

    # ID Mapping
    # 같은 방문지 이름이면 같은 id 부여하기
    if table_name == "visit":
        table = same_name_same_id(table)
        table = get_rating(table)

    # Feature 나누기 나홀로 여행 -> 인원수:1 / 자녀 동반 여부: false / 부모 동반여부: false
    if table_name == "user":
        table = split_feature(table)

    return table

## 16.5점 만점
def get_rating(table, weight_0 = 0.8, weight_1 = 1.0, weight_2 = 1.5):
    table['RATING'] = weight_0 * table['REVISIT_INTENTION'] + weight_1 * table['RCMDTN_INTENTION'] + weight_2 * table['DGSTFN']
    return table

def merge_table(visit, travel, user):
    merge_table = pd.merge(travel,user, how='inner',on='TRAVELER_ID')
    merge_table = pd.merge(visit, merge_table, how='inner',on='TRAVEL_ID')
    ## Reordering
    # merge_table = merge_table[['VISIT_ID', 'VISIT_AREA_ID', 'TRAVEL_ID', 'TRAVELER_ID', 'VISIT_AREA_NM', 'LOTNO_ADDR',
    #        'RESIDENCE_TIME_MIN', 'VISIT_AREA_TYPE_CD', 'REVISIT_YN', 'DGSTFN',
    #        'REVISIT_INTENTION', 'RCMDTN_INTENTION', 'RATING', 'VISIT_MM',
    #         'TRAVEL_NM', 'MVMN_NM', 'TRAVEL_PURPOSE', 'TRAVEL_MISSION', 'TRAVEL_MM',
    #        'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7', 'TRAVEL_STYL_8',
    #        'TRAVEL_STATUS_ACCOMPANY', 'NUMBER_OF_PERSON', 'CHILDREN', 'PARENTS']]
    merge_table = merge_table[['VISIT_ID', 'TRAVEL_ID', 'TRAVELER_ID', 'VISIT_AREA_NM', 'LOTNO_ADDR',
                               'RESIDENCE_TIME_MIN', 'VISIT_AREA_TYPE_CD', 'REVISIT_YN', 'DGSTFN',
                               'REVISIT_INTENTION', 'RCMDTN_INTENTION', 'RATING', 'VISIT_MM',
                               'MVMN_NM', 'TRAVEL_PURPOSE', 'TRAVEL_MISSION', 'TRAVEL_MM',
                               'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7',
                               'TRAVEL_STYL_8', 'NUMBER_OF_PERSON', 'CHILDREN', 'PARENTS']]
    # 위에서 일괄 드랍
    # merge_table = drop_columns(merge_table, ['VISIT_AREA_ID', 'TRAVEL_MM', 'TRAVEL_MM', 'TRAVEL_STATUS_ACCOMPANY'])

    return merge_table




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

    # divide train/test set
    train_data, test_data = train_test_split(user_visit_rating_matrix, test_size=0.2)
    test_set_mask = (test_data >= 10) & (test_data < 13)
    test_data[(test_set_mask >= 10) & (test_set_mask < 13)] = 0
    table = pd.concat([train_data, test_data])
    split_value = table.shape[0]-test_data.shape[0]

    ## collaborative filtering
    # user_based_result = user_based(table.copy(), np.array(test_data.index))
    # item_based_result = item_based(table.T.copy(), np.array(test_data.index))

    ## Model-based Filterting
    # singular_value_decomposition(train_data.T.copy())
    svd_result = singular_value_decomposition(table.copy(), test_data.index, n=1000)
    #
    # factorizer = MatrixFactorization(np.array(train_data), k=3, learning_rate=0.01, reg_param=0.01, epochs=300, verbose=True)
    factorizer = MatrixFactorization(np.array(user_visit_rating_matrix.copy()), k=3, learning_rate=0.01, reg_param=0.01, epochs=300, verbose=True)
    factorizer.fit()
    factorizer.print_results()

    # # # save the file
    table.to_csv("dataset/data_after_preprocessing/dataset.csv")
    processed_visit_data.to_csv("dataset/data_after_preprocessing/수도권_visit.csv")
    processed_travel_data.to_csv("dataset/data_after_preprocessing/수도권_travel.csv")
    processed_user_data.to_csv("dataset/data_after_preprocessing/수도권_user.csv")
    user_visit_rating_matrix.to_csv("dataset/data_after_preprocessing/rating_matrix.csv")

