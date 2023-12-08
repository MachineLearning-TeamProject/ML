import os 
import pandas as pd
import numpy as np

def get_info(region, visit_id):
    # open the file
    df = pd.read_csv(os.path.join("dataset", "data_after_preprocessing", region, "dataset.csv"))

    df = df.groupby("VISIT_ID")
    
    # 특정 방문지에 대한 평균 만족도
    mean_satisfaction = df.get_group(visit_id)["DGSTFN"].mean()

    # 특정 방문지에 대한 재방문 의향
    mean_revisit_intention = df.get_group(visit_id)["REVISIT_INTENTION"].mean()

    # 특정 방문지에 대한 추천 의향
    mean_recommendation_intention = df.get_group(visit_id)["RCMDTN_INTENTION"].mean()

    # 특정 방문지에 대한 리뷰 수
    review_count = df.get_group(visit_id)["RATING"].count()

    # Convert numpy.int64 values to regular Python integers
    mean_satisfaction = mean_satisfaction.item() if isinstance(mean_satisfaction, np.int64) else mean_satisfaction
    mean_revisit_intention = mean_revisit_intention.item() if isinstance(mean_revisit_intention, np.int64) else mean_revisit_intention
    mean_recommendation_intention = mean_recommendation_intention.item() if isinstance(mean_recommendation_intention, np.int64) else mean_recommendation_intention
    review_count = review_count.item() if isinstance(review_count, np.int64) else review_count
    
    return {"평균 만족도": mean_satisfaction, "평균 재방문 의향": mean_revisit_intention, "평균 추천 의향": mean_recommendation_intention, "리뷰 수": review_count}

