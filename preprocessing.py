import pandas as pd
import numpy as np

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
    table['VISIT_ID'] = table.groupby('VISIT_AREA_NM')['VISIT_AREA_ID'].transform('min')
    return table



# -------------------------
# Main Table Processing
# -------------------------
def process_table(table, table_name):
    # 날짜 합치기
    if table_name == "visit":
        table = merge_dates(table, "VISIT_START_YMD", "VISIT_MM")
    elif table_name == "travel":
        table = merge_dates(table, "TRAVEL_START_YMD", "VISIT_MM")

    # 각 테이블에 따른 추가 제거할 열
    if table_name == 'visit':
        columns_to_drop = ['VISIT_ORDER', 'ROAD_NM_ADDR', 'X_COORD', 'Y_COORD', 'ROAD_NM_CD','LOTNO_CD',
                                            'POI_ID','POI_NM', 'VISIT_CHC_REASON_CD', 'LODGING_TYPE_CD', 'SGG_CD', 'VISIT_START_YMD', 'VISIT_END_YMD',
                                            'VISIT_START_YMD', 'VISIT_END_YMD']
                                            
    elif table_name == 'travel':
        columns_to_drop = ['TRAVEL_PURPOSE', 'TRAVEL_PERSONA', 'TRAVEL_MISSION', 'TRAVEL_MISSION_CHECK',
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

    return table


if __name__ == "__main__":
    
    # open the file
    visit_data = pd.read_csv("dataset/수도권/tn_visit_area_info_방문지정보_A.csv")
    travel_data = pd.read_csv("dataset/수도권/tn_travel_여행_A.csv")
    user_data = pd.read_csv("dataset/수도권/tn_traveller_master_여행객 Master_A.csv")

    # preprocessing
    processed_visit_data = process_table(visit_data, "visit")
    processed_travel_data = process_table(travel_data, "travel")
    processed_user_data = process_table(user_data, "user")
    
    # # save the file
    processed_visit_data.to_csv("dataset/data_after_preprocessing/수도권_visit.csv")
    processed_travel_data.to_csv("dataset/data_after_preprocessing/수도권_travel.csv")
    processed_user_data.to_csv("dataset/data_after_preprocessing/수도권_user.csv")