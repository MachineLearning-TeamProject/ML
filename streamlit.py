import requests
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from streamlit_searchbox import st_searchbox
from streamlit_star_rating import st_star_rating
from streamlit_modal import Modal
import csv
import webbrowser

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

if 'go_to_info' not in st.session_state:
    st.session_state['go_to_info'] = True
    

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
    st.info(' 동서남북 어디로 떠나고 싶은가요?? \n\n 현재, 수도권, 동부권, 서부권, 제주도 및 도서지역이 지원됩니다.', icon="🧭")
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
    st.info(' 재밌었던 여행지, 별로였던 여행지 어디든 좋아요. \n\n 아래 검색창에서 검색 후, [가본 곳 추가하기] 버튼 클릭!', icon="🗺️")
    st.caption('가본 곳을 선택해주세요.')
    selected_value = st_searchbox(
        search_visit_area,
        key="visit_area_searchbox",
    )
    st.caption('선택된 곳들: ' + str(set(st.session_state['selected_values'])))

    if st.button('가본 곳 추가하기'):
        st.session_state['selected_values'] = st.session_state['selected_values'] + [selected_value]
        st.experimental_rerun() 
    
    if st.button('가본 곳 추가 완료'):
        st.session_state['select_voyage_stage'] = False
        st.session_state['rating_stage'] = True
        st.session_state['selected_values'] = list(set(st.session_state['selected_values']))
        st.experimental_rerun()

# -------------------------------
# Stage 3: 만족도 평가
# -------------------------------
if st.session_state['rating_stage'] == True:
    st.info(' 얼마나 만족스러운 여행이었나요? \n\n 간단한 설문에 답변해주세요!', icon="😆")
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
    st.info(' 이제 다 되었습니다! \n\n 아래 버튼을 클릭해 결과를 확인해보세요!', icon="😆")
    
    selected_model = st.selectbox('모델을 선택해주세요.',
            ('User-based filtering method', 'Memory-based filtering method', 'Content-based filtering method', 'SVD method', 'Matrix-Factorization method'),
            key = "key4")
    
    if st.button("비슷한 여행지 추천 받기"):
        with st.container(border = True):
            tmp_dict = {}
            # {st.session_state['selected_values'] : (st.session_state['revisit_rating'], st.session_state['recommend_rating'], st.session_state['satisfaction_rating'])} 형태로 딕셔너리로 저장되게 해 줘.
            for idx, voyage in enumerate(st.session_state['selected_values']):
                tmp_dict[voyage] = (int(st.session_state['revisit_rating'][idx]), int(st.session_state['recommend_rating'][idx]), int(st.session_state['satisfaction_rating'][idx]))
            
            if selected_model == 'User-based filtering method':

                st.markdown("# User-based filtering method")
                with st.spinner("추천 중입니다... ❤️"):
                    url = f"http://localhost:8080/user_based"
                    data = {
                        "region": st.session_state['selected_region'],
                        "user_visit": tmp_dict
                    }
                    response = requests.post(url, json=data) 
                st.write(response.json())
                
            elif selected_model == 'Memory-based filtering method':
                # [2] Memory based filtering method
                st.markdown("# Memory-based filtering method")
                with st.spinner("추천 중입니다... ❤️"):
                    url = f"http://localhost:8080/memory_based"
                    data = {
                        "region": st.session_state['selected_region'],
                        "user_visit": tmp_dict
                    }
                    response = requests.post(url, json=data) 
                if response.json()['detail'] == "Not Found":
                    st.write("관련 정보가 부족해, 아직 추천해 드릴 수 없습니다 😢")
                else: 
                    st.write(response.json())
                
            elif selected_model == 'Content-based filtering method':
                # [3] content based filtering method
                st.markdown("# Content-based filtering method")
                # print(st.session_state['visit_area_dict'])
                for visited_area_name in st.session_state['selected_values']:
                    with st.spinner("추천 중입니다... ❤️"):
                        visited_area_id = int(st.session_state['visit_area_dict'].get(visited_area_name))
                        st.session_state['visited_id'] = st.session_state['visited_id'] + [visited_area_id]
                        st.markdown("#### " + visited_area_name + '과 비슷한 여행지입니다.')
                        # FASTAPI인 http://localhost:8080/%EC%88%98%EB%8F%84%EA%B6%8C/content_based/3 호출
                        
                        url = f"http://localhost:8080/{st.session_state['selected_region']}/content_based/{visited_area_id}"
                        response = requests.get(url)
                        if len(response.json()) == 0:
                            st.write("관련 정보가 부족해, 아직 추천해 드릴 수 없습니다 😢")
                        else: 
                            st.write(response.json())
                    # st.write(response.json())

            elif selected_model == 'SVD method':
                # [4] model based filtering method - SVD
                st.markdown("# SVD method")
                with st.spinner("추천 중입니다... 10초 정도 소요됩니다 ❤️"):
                    url = f"http://localhost:8080/svd_based"
                    data = {
                        "region": st.session_state['selected_region'],
                        "user_visit": tmp_dict
                    }
                    response = requests.post(url, json=data) 
                st.write(response.json())
                
            elif selected_model == 'Matrix-Factorization method':
                # [5] model based filtering method - Matrix Factorization
                st.markdown("# Matrix Factorization method")
                with st.spinner("추천 중입니다... 20초 정도 소요됩니다 ❤️"):
                    url = f"http://localhost:8080/mf_based"
                    data = {
                        "region": st.session_state['selected_region'],
                        "user_visit": tmp_dict
                    }
                    response = requests.post(url, json=data) 
                st.write(response.json())

    st.divider()
    st.info(' 가고 싶은 곳이 생기셨나요? \n\n 아래 버튼을 클릭해 더 자세히 알아보세요!', icon="🚀")

    value_to_search = st_searchbox(
        search_visit_area,
        key="visit_area_searchbox",
    )

    if st.button('알아보기'):
        url = 'https://map.naver.com/p/search/' + str(value_to_search)
        webbrowser.open_new_tab(url)

    if st.button('다른 사용자들의 평가 보기'):  
        visited_area_id = int(st.session_state['visit_area_dict'].get(str(value_to_search)))
        st.markdown("#### " + value_to_search + '에 대한 다른 사용자들의 평가입니다. ')
        
        with st.container(border = True):
            url = f"http://localhost:8080/{st.session_state['selected_region']}/info/{visited_area_id}"
            with st.spinner("정보를 가져오고 있습니다... ❤️"):
                response = requests.get(url)
            st.markdown("### " + str(response.json()['리뷰 수']) + "명이 평가했습니다.")
            mean_satisfaction = float(response.json()['평균 만족도'])
            st_star_rating(label = "평균 만족도) " + str(round(mean_satisfaction, 2)), maxValue = 5, size = 30,defaultValue = round(mean_satisfaction), key = "rating", read_only = True,  customCSS = "div {font-size: 10px;}"  )
            
            mean_revisit_intention = float(response.json()['평균 재방문 의향'])
            st_star_rating(label = "평균 재방문 의향) " + str(round(mean_revisit_intention, 2)), maxValue = 5, size = 30,defaultValue = round(mean_revisit_intention), key = "rating2", read_only = True,  customCSS = "div {font-size: 10px;}"  )
            
            mean_recommend_intention = float(response.json()['평균 추천 의향'])
            st_star_rating(label = "평균 추천 의향) " + str(round(mean_recommend_intention, 2)), maxValue = 5, size = 30, defaultValue = round(mean_recommend_intention), key = "rating3", read_only = True,  customCSS = "div {font-size: 10px;}"  )
            
        
            
        
