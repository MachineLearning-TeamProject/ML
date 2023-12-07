import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_searchbox import st_searchbox
import csv
from predict_content_based import recommend_content 
from main import user_recommend


# Session State also supports attribute based syntax
# ----------------------------------------
# stage 변수 초기화
if 'select_region_stage' not in st.session_state:
    st.session_state['select_region_stage'] = True

if 'select_voyage_stage' not in st.session_state:
    st.session_state['select_voyage_stage'] = False

if 'rating_stage' not in st.session_state:
    st.session_state['rating_stage'] = False

if 'recommendation_stage' not in st.session_state:
    st.session_state['recommendation_stage'] = []

# ----------------------------------------
if 'selected_region' not in st.session_state:
    st.session_state['selected_region'] = ""

if 'visit_area_dict' not in st.session_state:
    st.session_state['visit_area_dict'] = {}

if 'visit_area_names' not in st.session_state:
    st.session_state['visit_area_names'] = []

if 'selected_values' not in st.session_state:
    st.session_state['selected_values'] = []

if 'satisfaction_rating' not in st.session_state:
    st.session_state['satisfaction_rating'] = []

if 'revisit_rating' not in st.session_state:
    st.session_state['revisit_rating'] = []

if 'recommend_rating' not in st.session_state:
    st.session_state['recommend_rating'] = []
    
if 'visited_id' not in st.session_state:
    st.session_state['visited_id'] = []

@st.cache_data()
def request_endpoint(url):
    response = requests.get(url)
    st.write(response.status_code)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return None

def setting_region(region):
    """
    Sets the visit_area_dict based on the specified region.

    Parameters:
    region (str): The region for which to set the visit_area_dict.

    Returns:
    list: A list of keys from the visit_area_dict.

    Raises:
    FileNotFoundError: If the specified region file is not found.
    """

    if region == '수도권':
        # visit_area_dict.csv 파일을 불러와서, key와 value 딕셔너리 생성

        with open("visit_area_dict_수도권.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[1]: row[0] for row in reader if row}
    elif region == '동부권':
        with open("visit_area_dict_동부권.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[1]: row[0] for row in reader if row}
    elif region == '서부권':
        with open("visit_area_dict_서부권.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[1]: row[0] for row in reader if row}
    elif region == '제주도 및 도서 지역':
        with open("visit_area_dict_제주도_및_도서_지역.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[1]: row[0] for row in reader if row}
        
    st.session_state['visit_area_dict'] = visit_area_dict
        # value값들만 리스트로 저장
    return list(visit_area_dict.keys())


merged_table = pd.read_csv("C:\ML\dataset\data_after_preprocessing\content_based_combined.csv", encoding="utf-8")



# ========================================
# 페이지 시작
# ========================================

st.title("Where to go?")

# -------------------------------
# Stage 1: 여행하고 싶은 지역 선택
# -------------------------------
if st.session_state['select_region_stage'] == True:
    region = st.selectbox(
                '여행하고 싶은 지역을 선택해주세요',
                ('수도권', '동부권', '서부권', '제주도 및 도서 지역')
            )
    if st.button("선택 완료"):
        st.session_state['select_region_stage'] = False
        st.session_state['select_voyage_stage'] = True
        st.session_state['selected_region'] = region
        st.session_state['visit_area_names'] = setting_region(region)
        st.experimental_rerun()


def search_visit_area(searchterm):
    return [visit_area_name for visit_area_name in st.session_state['visit_area_names'] if searchterm in visit_area_name]

# -------------------------------
# Stage 2: 가본 곳 선택
# -------------------------------
if st.session_state['select_voyage_stage'] == True:
    st.caption('가본 곳을 선택해주세요.')
    selected_value = st_searchbox(
        search_visit_area,
        key="visit_area_searchbox",
    )

    if st.button('가본 곳 추가하기'):
        st.session_state['selected_values'] = st.session_state['selected_values'] + [selected_value]
        # st.session_state['selected_values'] = st.session_state['selected_values'].append(selected_value)
        # selected_values.append(selected_value)
        st.text(st.session_state['selected_values'])

    if st.button('가본 곳 추가 완료'):
        st.session_state['select_voyage_stage'] = False
        st.session_state['rating_stage'] = True
        st.experimental_rerun()

# -------------------------------
# Stage 3: 만족도 평가
# -------------------------------
if st.session_state['rating_stage'] == True:
    for idx, voyage in enumerate(st.session_state['selected_values']):
        st.markdown("## " + voyage)
        globals()[f"option1_{idx}"] = st.selectbox(
            '만족도를 1~5점으로 평가해주세요.',
            ('1', '2', '3', '4', '5'),
            key = "key"+str(idx)
        )
        globals()[f"option2_{idx}"] = st.selectbox(
            '재방문 의향을 1~5점으로 평가해주세요.',
            ('1', '2', '3', '4', '5'),
            key = "key2"+str(idx)
        )
        globals()[f"option3_{idx}"] = st.selectbox(
            '추천 의향을 1~5점으로 평가해주세요.',
            ('1', '2', '3', '4', '5'),
            key = "key3"+str(idx)
        )
        
    if st.button("설문 완료"):
        st.session_state['rating_stage'] = False
        st.session_state['recommendation_stage'] = True
        st.session_state['satisfaction_rating'] = st.session_state['satisfaction_rating'] + [globals()[f"option1_{idx}"] for idx, voyage in enumerate(st.session_state['selected_values'])]
        st.session_state['revisit_rating'] = st.session_state['revisit_rating'] + [globals()[f"option2_{idx}"] for idx, voyage in enumerate(st.session_state['selected_values'])]
        st.session_state['recommend_rating'] = st.session_state['recommend_rating'] + [globals()[f"option3_{idx}"] for idx, voyage in enumerate(st.session_state['selected_values'])]
        st.experimental_rerun()
    
# -------------------------------
# Stage 4: 비슷한 여행지 추천
# -------------------------------
if st.session_state['recommendation_stage'] == True:
    
    # 비슷한 여행지 추천 받는 기능을 server.py에 fastapi 형태로 구현해줘
    # 그리고 그걸 여기서 불러와서 쓰면 될 듯
    # 그러면 여기서는 그냥 버튼 누르면 추천 받는 거로 해도 될 듯

    if st.button("비슷한 여행지 추천 받기"):
        tmp_dict = {}
        # {st.session_state['selected_values'] : (st.session_state['revisit_rating'], st.session_state['recommend_rating'], st.session_state['satisfaction_rating'])} 형태로 딕셔너리로 저장되게 해 줘.
        for idx, voyage in enumerate(st.session_state['selected_values']):
            tmp_dict[voyage] = (int(st.session_state['revisit_rating'][idx]), int(st.session_state['recommend_rating'][idx]), int(st.session_state['satisfaction_rating'][idx]))
        # with st.spinner("추천 중입니다... 30초 정도 소요됩니다 ❤️"):
            # recommend_list = user_recommend(area_code = 1, user_visit=tmp_dict)

        st.balloons()
        # User based filtering method
        st.markdown("# User-based filtering method")
        # st.text(recommend_list[0])
        st.divider()

        # Memory based filtering method
        st.markdown("# Memory-based filtering method")
        # st.text(recommend_list[1])
        st.divider()

        # content based filtering method
        st.markdown("# Content-based filtering method")
        # print(st.session_state['visit_area_dict'])
        for visited_area_name in st.session_state['selected_values']:
            
            visited_area_id = int(st.session_state['visit_area_dict'].get(visited_area_name))
            st.session_state['visited_id'] = st.session_state['visited_id'] + [visited_area_id]
            st.markdown("#### " + visited_area_name + '과 비슷한 여행지입니다.')
            # FASTAPI인 http://localhost:8080/%EC%88%98%EB%8F%84%EA%B6%8C/content_based/3 호출
            
            url = f"http://localhost:8080/{st.session_state['selected_region']}/content_based/{visited_area_id}"
            response = requests.get(url)
            st.write(response.json())

        st.divider()

        # model based filtering method
        st.markdown("# SVD method")
        # st.text(recommend_list[2])
        st.divider()

        st.markdown("# Matrix Factorization method")
        # st.text(recommend_list[3])
        st.divider()
    
        
    