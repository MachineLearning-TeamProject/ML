import pandas as pd
REGION = '도서지역'
dataset = pd.read_csv(f"dataset/data_after_preprocessing/{REGION}/dataset.csv")

# dataset에서 VISIT_AREA_NM만 뽑아서 리스트 형태로 visit_area_names에 저장
visit_area_names = dataset["VISIT_AREA_NM"].unique().tolist()

# VISIT_ID를 key로, VISIT_AREA_NM을 value로 하는 딕셔너리 생성
visit_area_dict = {}
for i in range(len(dataset)):
    visit_area_dict[dataset.iloc[i]["VISIT_ID"]] = dataset.iloc[i]["VISIT_AREA_NM"]

# 위 딕셔너리를 csv 파일로 저장 (utf-8로 인코딩) (csv 파일의 첫 번째 열은 key, 두 번째 열은 value로 저장)
import csv
with open(f"visit_area_dict_{REGION}.csv", "w", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["key", "value"])
    for key, value in visit_area_dict.items():
        writer.writerow([key, value])