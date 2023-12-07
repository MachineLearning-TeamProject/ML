import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from streamlit_searchbox import st_searchbox
import csv
from predict_content_based import recommend_content
from streamlit_star_rating import st_star_rating
# import SessionState

# Session State also supports attribute based syntax
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

@st.cache_data()
def request_endpoint(url):
    response = requests.get(url)
    st.write(response.status_code)
    if response.status_code == 200:
        return pd.DataFrame(response.json())
    return None

# visit_area_dict.csv 파일을 불러와서, key와 value 딕셔너리 생성

with open("visit_area_dict.csv", "r", encoding="utf-8") as f:
    next(f)
    reader = csv.reader(f)
    visit_area_dict = {row[1]: row[0] for row in reader if row}


# value값들만 리스트로 저장
visit_area_names = list(visit_area_dict.keys())


# visit_area_names에 searchterm이 포함된 단어들이 있으면, 그 단어들의 리스트 반환하는 함수
def search_visit_area(searchterm):
    return [visit_area_name for visit_area_name in visit_area_names if searchterm in visit_area_name]
    


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


# 가본 곳 여러 개 받기
# pass search function to searchbox
selected_value = st_searchbox(
    search_visit_area,
    key="visit_area_searchbox",
)

if st.button('가본 곳 추가하기'):
    st.session_state['selected_values'] = st.session_state['selected_values'] + [selected_value]
    # st.session_state['selected_values'] = st.session_state['selected_values'].append(selected_value)
    # selected_values.append(selected_value)
    st.text(st.session_state['selected_values'])

def function_to_run_on_click(value):
    st.write(f"**{value}** stars!")
# with st.echo(): 
#     stars = st_star_rating(label="별선택", maxValue=5, defaultValue=3, on_click=function_to_run_on_click)
#     st.write(stars)

if st.button('가본 곳 추가 완료'):
    st.session_state['rating_ok'] = True

if st.session_state['rating_ok'] == True:
    for idx, voyage in enumerate(st.session_state['selected_values']):
        st.text(voyage)
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
        st.session_state['satisfaction_rating'] = st.session_state['satisfaction_rating'] + [globals()[f"option1_{idx}"] for idx, voyage in enumerate(st.session_state['selected_values'])]
        st.session_state['revisit_rating'] = st.session_state['revisit_rating'] + [globals()[f"option2_{idx}"] for idx, voyage in enumerate(st.session_state['selected_values'])]
        st.session_state['recommend_rating'] = st.session_state['recommend_rating'] + [globals()[f"option3_{idx}"] for idx, voyage in enumerate(st.session_state['selected_values'])]
        st.experimental_rerun()
    
if st.button("비슷한 여행지 추천 받기"):
    st.markdown("# Content-based filtering method")
    for visited_area_name in st.session_state['selected_values']:
        print(visited_area_name)
        visited_area_id = int(visit_area_dict.get(visited_area_name))
        st.session_state['visited_id'] = st.session_state['visited_id'] + [visited_area_id]
        st.markdown("#### " + visited_area_name + '과 비슷한 여행지입니다.')
        result_df = recommend_content(merged_table, visited_area_id)
        for idx, row in result_df.iterrows():    
            st.text(row['방문지명'])
    st.divider()
        
    