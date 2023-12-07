import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_searchbox import st_searchbox
import csv
from predict_content_based import recommend_content 

# Session State also supports attribute based syntax

if 'select_region_stage' not in st.session_state:
    st.session_state['select_region_stage'] = True

if 'selected_region' not in st.session_state:
    st.session_state['selected_region'] = ""

if 'visit_area_dict' not in st.session_state:
    st.session_state['visit_area_dict'] = {}

if 'visit_area_names' not in st.session_state:
    st.session_state['visit_area_names'] = []

if 'select_voyage_stage' not in st.session_state:
    st.session_state['select_voyage_stage'] = []

if 'selected_values' not in st.session_state:
    st.session_state['selected_values'] = []

if 'rating_ok' not in st.session_state:
    st.session_state['rating_ok'] = False

if 'satisfaction_rating' not in st.session_state:
    st.session_state['satisfaction_rating'] = []

if 'revisit_rating' not in st.session_state:
    st.session_state['revisit_rating'] = []

if 'recommend_rating' not in st.session_state:
    st.session_state['recommend_rating'] = []
    
if 'visited_id' not in st.session_state:
    st.session_state['visited_id'] = []

if 'recommendation_stage' not in st.session_state:
    st.session_state['recommendation_stage'] = []

@st.cache_data()
def request_endpoint(url):
    response = requests.get(url)
    st.write(response.status_code)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return None

def setting_region(region):
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


# -----------------
# 페이지 이름
# -----------------
page1_name = "Item-based collaborative filtering"
page2_name = "User-based collaborative filtering"
page3_name = "Content-based filtering"

# -----------------
# 슬라이드 바
# -----------------
# st.sidebar.subheader("SageMaker Endpoint")

# page = st.sidebar.selectbox("페이지 선택", [page1_name, page2_name, page3_name])

# api_url = st.sidebar.text_input('fast api', value="43.202.112.5:8000")

# user_id = st.sidebar.text_input('user id', value=1)

# threshold = st.sidebar.slider('prediction threshold', 0.0, 1.0, 0.2)



# -----------------
# 타이틀
# -----------------

st.title("Where to go?")

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

# 가본 곳 여러 개 받기
# pass search function to searchbox
# visit_area_names에 searchterm이 포함된 단어들이 있으면, 그 단어들의 리스트 반환하는 함수
def search_visit_area(searchterm):
    return [visit_area_name for visit_area_name in st.session_state['visit_area_names'] if searchterm in visit_area_name]

if st.session_state['select_voyage_stage'] == True:
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
        st.session_state['rating_ok'] = True
        st.experimental_rerun()

if st.session_state['rating_ok'] == True:
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
        st.session_state['rating_ok'] = False
        st.session_state['recommendation_stage'] = True
        st.session_state['satisfaction_rating'] = st.session_state['satisfaction_rating'] + [globals()[f"option1_{idx}"] for idx, voyage in enumerate(st.session_state['selected_values'])]
        st.session_state['revisit_rating'] = st.session_state['revisit_rating'] + [globals()[f"option2_{idx}"] for idx, voyage in enumerate(st.session_state['selected_values'])]
        st.session_state['recommend_rating'] = st.session_state['recommend_rating'] + [globals()[f"option3_{idx}"] for idx, voyage in enumerate(st.session_state['selected_values'])]
        st.experimental_rerun()
    
if st.session_state['recommendation_stage'] == True:
    if st.button("비슷한 여행지 추천 받기"):
        st.markdown("# User-based filtering method")

        st.divider()


        st.markdown("# Memory-based filtering method")

        st.divider()


        st.markdown("# Content-based filtering method")
        # print(st.session_state['visit_area_dict'])
        for visited_area_name in st.session_state['selected_values']:
            
            visited_area_id = int(st.session_state['visit_area_dict'].get(visited_area_name))
            st.session_state['visited_id'] = st.session_state['visited_id'] + [visited_area_id]
            st.markdown("#### " + visited_area_name + '과 비슷한 여행지입니다.')
            result_df = recommend_content(merged_table, visited_area_id, st.session_state['visit_area_dict'])
            for idx, row in result_df.iterrows():    
                st.text(row['방문지명'])
        st.divider()
    
        
    