import pandas as pd
import numpy as np

# Load the data
data = pd.read_csv('/content/rating movies S.csv')

# Preprocess the data
# Remove duplicates based on 'userId' and 'rating'
data = data.drop_duplicates(subset=['userId', 'rating'])

# Convert 'rating' to integer after filling NaNs with 0 if necessary
data['rating'] = data['rating'].fillna(0).astype(int)

# Convert 'timestamp' from UNIX format to datetime
data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')

# Create the user-item matrix without filling NaNs after pivoting
user_item_matrix = data.pivot_table(index='userId', columns='movieId', values='rating')

# Fill all NaN values with random integers between 1 and 5 for demonstration
user_item_matrix = user_item_matrix.applymap(lambda x: np.random.randint(1, 6) if pd.isna(x) else x)

# Select a random subset (8x8) of the matrix
random_user_ids = np.random.choice(user_item_matrix.index, 8, replace=False)
random_movie_ids = np.random.choice(user_item_matrix.columns, 8, replace=False)
user_item_matrix_subset = user_item_matrix.loc[random_user_ids, random_movie_ids]

# Randomly introduce 3-4 NaN values into the subset matrix
num_nan_values = np.random.choice([3, 4])  # Randomly choose between 3-4 NaN values
nan_indices = [(np.random.choice(user_item_matrix_subset.index), np.random.choice(user_item_matrix_subset.columns)) for _ in range(num_nan_values)]
for row, col in nan_indices:
    user_item_matrix_subset.at[row, col] = np.nan

# Display the subset with the title
print("8x8 User-Item Matrix with 3-4 Random Null Values:")
print(user_item_matrix_subset)