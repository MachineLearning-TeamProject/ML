from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
from predict_content_based import recommend_content

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
    
    
    
    



if __name__ == '__main__':
    
    uvicorn.run("server:app", host='0.0.0.0', port=8080, workers=1)  # reload=False 권장
