from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import csv
import pandas as pd
import os
from preprocessing import process_table, merge_table, get_rating_
import numpy as np
from memory_based import user_based, item_based
from model_based import MatrixFactorization, singular_value_decomposition
from evaluation import recommend
from main import read_data

def data_preprocessing(area_code, user_visit):
    """
    Preprocesses the data for recommendation system.

    Args:
        area_code (str): The area code.
        user_visit (dict): Dictionary containing user visit data.

    Returns:
        tuple: A tuple containing the rating matrix, rating matrix index, user visit rating matrix, and dataset.
    """
    def add_user(dic, user_visit_rating_matrix, dataset):
        user_id = 'z000001'
        user_visit_rating_matrix = user_visit_rating_matrix.T
        user_visit_rating_matrix[user_id] = np.zeros(user_visit_rating_matrix.shape[0])
        for visit_nm in dic.keys():
            row_id = np.array(dataset[dataset['VISIT_AREA_NM']==visit_nm]['VISIT_ID'])[0]
            user_visit_rating_matrix.loc[row_id, user_id]= get_rating_(list(dic[visit_nm]))
        return user_visit_rating_matrix.T, [user_id]

    # open the file
    visit_data, travel_data, user_data = read_data(area_code)

    # preprocessing
    processed_visit_data = process_table(visit_data, "visit")
    processed_travel_data = process_table(travel_data, "travel")
    processed_user_data = process_table(user_data, "user")
    dataset = merge_table(processed_visit_data, processed_travel_data, processed_user_data)

    # row : User, column : item
    user_visit_rating_matrix = dataset.pivot_table(index='TRAVELER_ID', columns='VISIT_ID', values='RATING').fillna(0)

    user_visit_rating_matrix, user_id = add_user(user_visit, user_visit_rating_matrix, dataset)
    rating_matrix = user_visit_rating_matrix
    rating_matrix_index = user_id

    return rating_matrix, rating_matrix_index, user_visit_rating_matrix, dataset


def recommend_content(region, visit_id):
    """
    Recommends content based on the given region and visit ID.

    Parameters:
    region (str): The region for which content recommendations are needed.
    visit_id (int): The ID of the visit for which content recommendations are needed.

    Returns:
    pandas.DataFrame: A DataFrame containing the recommended content with visit area names and similarity scores.
    """
    # Read the dataset
    if region == '수도권':
        table = pd.read_csv('C:\ML\dataset\data_after_preprocessing\수도권\content_based_combined.csv')
        with open("visit_area_dict_수도권.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[0]: row[1] for row in reader if row}
    elif region == '동부권':
        table = pd.read_csv('C:\ML\dataset\data_after_preprocessing\동부권\content_based_combined.csv')
        with open("visit_area_dict_동부권.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[0]: row[1] for row in reader if row}
    elif region == '서부권': 
        table = pd.read_csv('C:\ML\dataset\data_after_preprocessing\서부권\content_based_combined.csv')
        with open("visit_area_dict_서부권.csv", "r", encoding="utf-8") as f:
            next(f)
            reader = csv.reader(f)
            visit_area_dict = {row[0]: row[1] for row in reader if row}
    elif region == '제주도 및 도서 지역':
        table = pd.read_csv('C:\ML\dataset\data_after_preprocessing\도서지역\content_based_combined.csv')
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

def recommend_user(area_code, user_visit):
    """
    Recommends users based on the given area code and user visit data.

    Parameters:
    area_code (str): The area code for which users are recommended.
    user_visit (list): The user visit data.

    Returns:
    list: The list of recommended users.
    """
    rating_matrix, rating_matrix_index, user_visit_rating_matrix, dataset = data_preprocessing(area_code, user_visit)

    # collaborative filtering
    user_based_result = user_based(rating_matrix.copy(), np.array(rating_matrix_index))
    recommend_list = recommend(dataset, user_visit_rating_matrix.T[rating_matrix_index], user_based_result, rating_matrix_index, 8.25)

    return recommend_list

def recommend_item(area_code, user_visit):
    """
    Recommends items based on collaborative filtering.

    Parameters:
    area_code (str): The area code.
    user_visit (list): The user visit data.

    Returns:
    list: The recommended item list.
    """
    rating_matrix, rating_matrix_index, user_visit_rating_matrix, dataset = data_preprocessing(area_code, user_visit)

    # collaborative filtering
    item_based_result = item_based(rating_matrix.T.copy(), np.array(rating_matrix_index))
    recommend_list = recommend(dataset, user_visit_rating_matrix.T[rating_matrix_index], item_based_result, rating_matrix_index, 8.25)
    
    return recommend_list


def recommend_svd(area_code, user_visit):
    """
    Recommends items using Singular Value Decomposition (SVD) based filtering.

    Args:
        area_code (str): The area code for which recommendations are generated.
        user_visit (list): The user visit data.

    Returns:
        list: The list of recommended items.
    """
    rating_matrix, rating_matrix_index, user_visit_rating_matrix, dataset = data_preprocessing(area_code, user_visit)

    # model_based filtering
    svd_result = singular_value_decomposition(rating_matrix.copy(), rating_matrix_index, n=1000)

    recommend_list = recommend(dataset, user_visit_rating_matrix.T[rating_matrix_index], svd_result, rating_matrix_index, 0.2)
    
    return recommend_list

def recommend_mf(area_code, user_visit):
    """
    Recommends items to a user based on matrix factorization.

    Args:
        area_code (str): The area code for which recommendations are generated.
        user_visit (list): The user's visit history.

    Returns:
        list: The recommended list of items.
    """
    rating_matrix, rating_matrix_index, user_visit_rating_matrix, dataset = data_preprocessing(area_code, user_visit)

    factorizer = MatrixFactorization(rating_matrix.copy(), k=3, learning_rate=0.01, reg_param=0.01, epochs=50, verbose=False)
    factorizer.load_array()
    factorizer.fit()
    mf_result = factorizer.test(rating_matrix_index)
    recommend_list = recommend(dataset, user_visit_rating_matrix.T[rating_matrix_index], mf_result, rating_matrix_index, 8.25)
    
    return recommend_list