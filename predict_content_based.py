import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv

# def recommend_content(table, visited_id, visit_area_dict):
#     visit_area_dict = {v:k for k,v in visit_area_dict.items()}

    
#     # Drop rows where 'TAG' is NaN or 'np'
#     table = table.dropna(subset=['TAG'])
#     table = table[table['TAG'] != 'np']

#     # Apply CountVectorizer to convert TAG into a matrix of token counts
#     count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
#     tag_matrix = count_vectorizer.fit_transform(table['TAG'])

#     # Calculate cosine similarity between items (VISIT_IDs)
#     cosine_sim = cosine_similarity(tag_matrix, tag_matrix)

#     # Function to get recommendations based on the cosine similarity
#     def get_recommendations(visit_id, cosine_sim=cosine_sim, table=table):
#         # Get the index of the visit_id using .loc
#         idx = table.loc[table['VISIT_ID'] == visit_id].index
        
        
#         if not idx.empty:
#             idx = idx[0]  # Access the first index if there are multiple matches
            
#             # Get the pairwise similarity scores
#             sim_scores = list(enumerate(cosine_sim[idx]))

#             # Sort the visits based on similarity scores
#             sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
#             # Get the top 5 most similar visits (excluding itself)
#             top_similar_visits = [item[0] for item in sim_scores[1:11]]
            
#             # Create a DataFrame with recommendations and corresponding cosine similarity
            
#             recommendations = table.iloc[top_similar_visits]  # Assuming the last column is 'TAG'
            
#             similarity_scores = [item[1] for item in sim_scores[1:11]]
            
#             recommendations['유사도'] = similarity_scores
            
#             # recommendations['VISIT_ID'].map(visit_area_dict)
#             recommendations['VISIT_ID'] = recommendations['VISIT_ID'].astype(str)
#             recommendations['VISIT_ID'].map(visit_area_dict)
            
        
#             # Add visit area names to recommendations using visit_area_dict
#             recommendations['방문지명'] = recommendations['VISIT_ID'].map(visit_area_dict)
            
    
#             print(recommendations.head())

#             return recommendations[['방문지명']]
#         else:
#             print(f"VISIT_ID {visit_id} not found in the dataset.")
#             return pd.DataFrame()

#     # idx = table.loc[table['VISIT_ID'] == visited_id].index
#     recommendations = get_recommendations(visited_id)
    
#     return recommendations

def recommend_content(region, visit_id):

    if region == '수도권':
        table = pd.read_csv('C:\ML\dataset\data_after_preprocessing\content_based_combined.csv')
        with open("visit_area_dict_수도권.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[0]: row[1] for row in reader if row}
    elif region == '동부권':
        table = pd.read_csv('C:\ML\dataset\data_after_preprocessing\content_based_combined.csv')
        with open("visit_area_dict_동부권.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[0]: row[1] for row in reader if row}
    elif region == '서부권':
        table = pd.read_csv('C:\ML\dataset\data_after_preprocessing\content_based_combined.csv')
        with open("visit_area_dict_서부권.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[0]: row[1] for row in reader if row}
    elif region == '제주도 및 도서 지역':
        table = pd.read_csv('C:\ML\dataset\data_after_preprocessing\content_based_combined.csv')
        with open("visit_area_dict_제주도_및_도서_지역.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[0]: row[1] for row in reader if row}
    
    # Drop rows where 'TAG' is NaN or 'np'
    table = table.dropna(subset=['TAG'])
    table = table[table['TAG'] != 'np']
    
    # Apply CountVectorizer to convert TAG into a matrix of token counts
    count_vectorizer = CountVectorizer(tokenizer=lambda x: x.split(', '))
    tag_matrix = count_vectorizer.fit_transform(table['TAG'])

    # Calculate cosine similarity between items (VISIT_IDs)
    cosine_sim = cosine_similarity(tag_matrix, tag_matrix)

    # Function to get recommendations based on the cosine similarity
    def get_recommendations(visit_id, cosine_sim=cosine_sim, table=table):
        # Get the index of the visit_id using .loc
        
        idx = table.loc[table['VISIT_ID'] == visit_id].index
        
        if not idx.empty:
            idx = idx[0]  # Access the first index if there are multiple matches
            
            # Get the pairwise similarity scores
            sim_scores = list(enumerate(cosine_sim[idx]))

            # Sort the visits based on similarity scores
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            
            # Get the top 5 most similar visits (excluding itself)
            top_similar_visits = [item[0] for item in sim_scores[1:11]]
            
            # Create a DataFrame with recommendations and corresponding cosine similarity
            
            recommendations = table.iloc[top_similar_visits]  # Assuming the last column is 'TAG'
            
            similarity_scores = [item[1] for item in sim_scores[1:11]]
            
            recommendations['유사도'] = similarity_scores
            
            # recommendations['VISIT_ID'].map(visit_area_dict)
            recommendations['VISIT_ID'] = recommendations['VISIT_ID'].astype(str)
            recommendations['VISIT_ID'].map(visit_area_dict)
            
        
            # Add visit area names to recommendations using visit_area_dict
            recommendations['방문지명'] = recommendations['VISIT_ID'].map(visit_area_dict)
            
    
            print(recommendations.head())

            return recommendations[['방문지명']]
        else:
            print(f"VISIT_ID {visit_id} not found in the dataset.")
            return pd.DataFrame()

    # idx = table.loc[table['VISIT_ID'] == visited_id].index
    recommendations = get_recommendations(visit_id, table=table)
    
    return recommendations