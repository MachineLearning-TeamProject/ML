from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
from predict import recommend_content, recommend_user, recommend_item, recommend_svd, recommend_mf

app = FastAPI()

@app.get("/")
async def hello():
    return {"hello":"world"}

# streamlit.py에서 content_based filtering 써서 비슷한 여행지 추천해주는 기능을 api로 구현
@app.get("/{region}/content_based/{visit_id}")
async def content_based(region:str, visit_id: int):
    print(region)
    print(visit_id)
    return recommend_content(region, visit_id)

@app.get("/{region}/content_based/{user_visit}")
async def content_based(region:str, user_visit: dict):
    area = {"수도권":1, "동부권": 2, "서부권": 3, "제주도 및 도서 지역": 4}
    area_code = area.get(region)

    print(region)
    print(user_visit)
    return recommend_user(area_code, user_visit)
    
    
    
    



if __name__ == '__main__':
    
    uvicorn.run("server:app", host='0.0.0.0', port=8080, workers=1)  # reload=False 권장
