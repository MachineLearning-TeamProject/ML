import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

# -----------------
# Utility Functions
# -----------------
# merge feature
def merge_dates(table, date_column, new_column_name):
    table[new_column_name] = table[date_column].str.slice(start=5, stop=7)
    return table

# drop column
def drop_columns(table, columns_to_drop):
    return table.drop(columns=columns_to_drop)

def replace_values(table, column_name, value_mapping):
    return table.replace({column_name: value_mapping})

def drop_null_values(table, columns_to_drop_na):
    return table.dropna(subset=columns_to_drop_na)

# Drop wrong data
def drop_non_spots(table):
    num_of_rows_before = len(table)
    table_with_only_touristSpots = table[(table['VISIT_AREA_TYPE_CD'] < 9) | (table['VISIT_AREA_TYPE_CD'] == 13)]
    num_of_rows_after = len(table_with_only_touristSpots)
    print(f"방문지 유형 코드가 1~8 혹은 13이 아닌 방문지 {num_of_rows_before - num_of_rows_after}개를 삭제했습니다.")
    return table_with_only_touristSpots

# Deriving Features(VISIT_ID) from Existing Features(VISIT_AREA_ID, TRAVEL_ID)
def same_name_same_id(table):
    # Groupby to VISIT_AREA_NM, and give visit_id from the front
    table['VISIT_ID'] = table.groupby('VISIT_AREA_NM').ngroup() + 1
    is_unique = not table['VISIT_ID'].duplicated().any()
    print(f"All values in 'VISIT_ID' are unique: {is_unique}")

    return table

def make_user_feature(table, ids, feature1, feature2, feature3):
    table.loc[ids, "NUMBER_OF_PERSON"] = feature1
    table.loc[ids, "CHILDREN"] = feature2
    table.loc[ids, "PARENTS"] = feature3
    return table

# Feature creation
# Deriving Features(NUMBER_OF_PERSON, CHILDREN, PARENTS) from Existing Features(TRAVEL_STATUS_ACCOMPANY)
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
    return table

# drop noise in VISIT_AREA_NM column
def drop_noise(table):
    table = table[~table['VISIT_AREA_NM'].str.endswith("역")]
    table = table[~table['VISIT_AREA_NM'].str.endswith("학교")]
    table = table[~table['VISIT_AREA_NM'].str.endswith("점")]

    return table

# -------------------------
# Main Table Processing
# -------------------------
def process_table(table, table_name):
    # Feature Creation
    if table_name == "visit":
        table = merge_dates(table, "VISIT_START_YMD", "VISIT_MM")
    elif table_name == "travel":
        table = merge_dates(table, "TRAVEL_START_YMD", "TRAVEL_MM")

    # Columns to be removed according to each table
    if table_name == 'visit':
        columns_to_drop = ['VISIT_ORDER', 'ROAD_NM_ADDR', 'X_COORD', 'Y_COORD', 'ROAD_NM_CD', 'LOTNO_CD',
                           'POI_ID', 'POI_NM', 'VISIT_CHC_REASON_CD', 'LODGING_TYPE_CD', 'SGG_CD', 'VISIT_START_YMD',
                           'VISIT_END_YMD',
                           'VISIT_START_YMD', 'VISIT_END_YMD']

    elif table_name == 'travel':
        columns_to_drop = ['TRAVEL_PERSONA', 'TRAVEL_MISSION_CHECK',
                           'TRAVEL_START_YMD', 'TRAVEL_END_YMD']
    elif table_name == 'user':
        columns_to_drop = ['RESIDENCE_SGG_CD', 'GENDER', 'AGE_GRP', 'EDU_NM', 'EDU_FNSH_SE', 'MARR_STTS', 'FAMILY_MEMB',
                           'JOB_NM',
                           'JOB_ETC', 'INCOME', 'HOUSE_INCOME', 'TRAVEL_TERM', 'TRAVEL_NUM', 'TRAVEL_LIKE_SIDO_1',
                           'TRAVEL_LIKE_SGG_1',
                           'TRAVEL_LIKE_SIDO_2', 'TRAVEL_LIKE_SGG_2', 'TRAVEL_LIKE_SIDO_3', 'TRAVEL_LIKE_SGG_3',
                           'TRAVEL_STYL_2',
                           'TRAVEL_STYL_4', 'TRAVEL_STATUS_RESIDENCE', 'TRAVEL_STATUS_DESTINATION', 'TRAVEL_STATUS_YMD',
                           'TRAVEL_MOTIVE_1'
            , 'TRAVEL_MOTIVE_2', 'TRAVEL_MOTIVE_3', 'TRAVEL_COMPANIONS_NUM']
    # drop column
    table = drop_columns(table, columns_to_drop)

    # VISIT_AREA_TYPE_CD filtering
    if table_name == "visit":
        table = drop_non_spots(table)
        table = drop_noise(table)

    # REVISIT_YN encoding
    if 'REVISIT_YN' in table.columns:
        table = replace_values(table, 'REVISIT_YN', {'Y': 1, 'N': 0})

    # Drop null values
    if 'REVISIT_YN' in table.columns:
        table = drop_null_values(table, ['REVISIT_YN', 'DGSTFN', 'REVISIT_INTENTION', 'RCMDTN_INTENTION'])

    # ID Mapping
    if table_name == "visit":
        table = same_name_same_id(table)
        table = get_rating(table)

    # Split Feature
    if table_name == "user":
        table = split_feature(table)

    return table


## calculate rating using weight sum
def get_rating(table, weight_0=0.8, weight_1=1.0, weight_2=1.5):
    table['RATING'] = weight_0 * table['REVISIT_INTENTION'] + weight_1 * table['RCMDTN_INTENTION'] + weight_2 * table[
        'DGSTFN']
    return table


## calculate rating using weight sum
def get_rating_(value, weight_0=0.8, weight_1=1.0, weight_2=1.5):
    return weight_0 * value[0] + weight_1 * value[0] + weight_2 * value[0]


# Merge three tables(visit table, travel table, user table).
def merge_table(visit, travel, user):
    merge_table = pd.merge(travel, user, how='inner', on='TRAVELER_ID')
    merge_table = pd.merge(visit, merge_table, how='inner', on='TRAVEL_ID')
    ## Reordering
    merge_table = merge_table[['VISIT_ID', 'TRAVEL_ID', 'TRAVELER_ID', 'VISIT_AREA_NM', 'LOTNO_ADDR',
                               'RESIDENCE_TIME_MIN', 'VISIT_AREA_TYPE_CD', 'REVISIT_YN', 'DGSTFN',
                               'REVISIT_INTENTION', 'RCMDTN_INTENTION', 'RATING', 'VISIT_MM',
                               'MVMN_NM', 'TRAVEL_PURPOSE', 'TRAVEL_MISSION', 'TRAVEL_MM',
                               'TRAVEL_STYL_1', 'TRAVEL_STYL_3', 'TRAVEL_STYL_5', 'TRAVEL_STYL_6', 'TRAVEL_STYL_7',
                               'TRAVEL_STYL_8', 'NUMBER_OF_PERSON', 'CHILDREN', 'PARENTS']]
    return merge_table
