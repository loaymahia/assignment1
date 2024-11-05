import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

# Sample user-item rating matrix (example matrix, replace with your data)
data = {
    'movieId': [481, 2500, 6537, 2539, 4995, 1213, 2321, 72],
    'user_400': [4, 3, 5, 4, 5, 2, 1, np.nan],
    'user_268': [3, 1, 5, 4, 3, 1, np.nan, 3],
    'user_370': [3, 4, 1, 2, 5, 5, 4, 2],
    'user_205': [4, 5, 3, 3, 4, 1, 4, np.nan],
    'user_634': [1, 2, 4, 1, 2, 5, 2, 3],
    'user_560': [1, 5, 3, 1, 1, 5, 4, 4],
    'user_74': [4, 1, 5, 1, 3, 5, 4, 3],
    'user_279': [1, 1, 1, 3, 3, 5, np.nan, 1]
}
df = pd.DataFrame(data).set_index('movieId')

# Calculate Cosine Similarity for users
cosine_sim = cosine_similarity(df.T.fillna(0))
cosine_sim_df = pd.DataFrame(cosine_sim, index=df.columns, columns=df.columns)

# Function to calculate Pearson correlation for each user pair
def pearson_corr_matrix(df):
    user_corr = pd.DataFrame(index=df.columns, columns=df.columns)
    for u in df.columns:
        for v in df.columns:
            user_corr.loc[u, v] = pearsonr(df[u].fillna(0), df[v].fillna(0))[0]
    return user_corr

pearson_sim_df = pearson_corr_matrix(df)

# Prediction function for User-Based Collaborative Filtering
def predict_user_based(user, item_id, similarity_matrix, ratings_df, method='cosine'):
    if method == 'cosine':
        sim_scores = similarity_matrix[user]
    elif method == 'pearson':
        sim_scores = similarity_matrix[user]
    
    rated_by_other_users = ratings_df.loc[item_id].dropna()
    weighted_sum = np.dot(sim_scores[rated_by_other_users.index], rated_by_other_users)
    sum_of_weights = np.abs(sim_scores[rated_by_other_users.index]).sum()
    
    return weighted_sum / sum_of_weights if sum_of_weights != 0 else np.nan

# Generate predictions for a specific user and item using both similarity measures
user = 'user_400'
item_id = 6537  # movie ID example

cosine_pred = predict_user_based(user, item_id, cosine_sim_df, df, method='cosine')
pearson_pred = predict_user_based(user, item_id, pearson_sim_df, df, method='pearson')

print(f"Predicted rating for {user} on item {item_id} using Cosine Similarity: {cosine_pred}")
print(f"Predicted rating for {user} on item {item_id} using Pearson Correlation: {pearson_pred}")

# Top-N Recommendations
def top_n_recommendations(user, n, similarity_matrix, ratings_df, method='cosine'):
    preds = []
    for item_id in ratings_df.index:
        if pd.isna(ratings_df.loc[item_id, user]):  # Only predict for items not yet rated
            pred_rating = predict_user_based(user, item_id, similarity_matrix, ratings_df, method)
            preds.append((item_id, pred_rating))
    preds = sorted(preds, key=lambda x: x[1], reverse=True)
    return preds[:n]

top_n_cosine = top_n_recommendations(user, 3, cosine_sim_df, df, method='cosine')
top_n_pearson = top_n_recommendations(user, 3, pearson_sim_df, df, method='pearson')

print(f"Top-N Recommendations for {user} using Cosine Similarity: {top_n_cosine}")
print(f"Top-N Recommendations for {user} using Pearson Correlation: {top_n_pearson}")