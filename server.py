from fastapi import FastAPI
import uvicorn
from predict import recommend_content, recommend_user, recommend_item, recommend_svd, recommend_mf
from info import get_info
from pydantic import BaseModel

app = FastAPI()

# Endpoint for content-based recommendation
@app.get("/{region}/content_based/{visit_id}")
async def content_based(region: str, visit_id: str):
    print(region)
    print(visit_id)
    result = recommend_content(region, int(visit_id))
    return result

# Data model for user-based recommendation
class History(BaseModel):
    region: str
    user_visit: dict

# Endpoint for user-based recommendation
@app.post("/user_based")
async def user_based(data_request: History):
    area = {"수도권": 1, "동부권": 2, "서부권": 3, "제주도 및 도서 지역": 4}
    area_code = area.get(data_request.region)

    print(data_request.region)
    print(data_request.user_visit)
    return recommend_user(area_code, data_request.user_visit)

# Endpoint for item-based recommendation
@app.post("/item_based")
async def item_based(data_request: History):
    area = {"수도권": 1, "동부권": 2, "서부권": 3, "제주도 및 도서 지역": 4}
    area_code = area.get(data_request.region)

    print(data_request.region)
    print(data_request.user_visit)
    return recommend_item(area_code, data_request.user_visit)

# Endpoint for SVD-based recommendation
@app.post("/svd_based")
async def svd_based(data_request: History):
    area = {"수도권": 1, "동부권": 2, "서부권": 3, "제주도 및 도서 지역": 4}
    area_code = area.get(data_request.region)

    print(data_request.region)
    print(data_request.user_visit)
    return recommend_svd(area_code, data_request.user_visit)

# Endpoint for matrix factorization-based recommendation
@app.post("/mf_based")
async def mf_based(data_request: History):
    area = {"수도권": 1, "동부권": 2, "서부권": 3, "제주도 및 도서 지역": 4}
    area_code = area.get(data_request.region)

    print(data_request.region)
    print(data_request.user_visit)
    return recommend_mf(area_code, data_request.user_visit)

# Endpoint for retrieving information
@app.get("/{region}/info/{visit_id}")
async def info(region: str, visit_id: str):
    result = get_info(region, int(visit_id))
    return result

if __name__ == '__main__':
    uvicorn.run("server:app", host='0.0.0.0', port=8080, workers=1)  # reload=False recommended
