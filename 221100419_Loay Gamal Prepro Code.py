import pandas as pd
from google.colab import files

# Load the dataset
df = pd.read_csv('/content/rating movies S.csv')

# Display the first few rows of the original dataset
print("Original Dataset:")
print(df.head())

# Step 1: Remove duplicates based on all columns
df.drop_duplicates(keep='first', inplace=True)
print("\nDataset after removing duplicates:")
print(df.head())

# Step 2: Handling null values
# Check for null values
null_counts = df.isnull().sum()
print("\nNull value counts before handling:")
print(null_counts)

# Option 1: Drop rows with any null values
df.dropna(inplace=True)

# Option 2: Alternatively, you can fill null values (uncomment if needed)
# df['rating'].fillna(df['rating'].mean(), inplace=True)

# Display the null value counts after handling
null_counts_after = df.isnull().sum()
print("\nNull value counts after handling:")
print(null_counts_after)

# Step 3: Remove the timestamp column if it's not needed
if 'timestamp' in df.columns:
    df.drop(columns=['timestamp'], inplace=True)
    print("\nTimestamp column removed.")

# Step 4: Standardize data types
# Convert ratings to integers if 'rating' column exists
if 'rating' in df.columns:
    df['rating'] = df['rating'].astype(int)

# Step 5: Encoding categorical variables (if applicable)
# Assuming you have a 'product_category' column for one-hot encoding
if 'product_category' in df.columns:
    df = pd.get_dummies(df, columns=['product_category'], drop_first=True)
    print("\nCategorical variables encoded.")

# Step 6: Normalize ratings (if needed)
if 'rating' in df.columns:
    df['rating'] = (df['rating'] - df['rating'].min()) / (df['rating'].max() - df['rating'].min())
    print("\nRatings normalized.")

# Display the final preprocessed dataset
print("\nFinal Preprocessed Dataset:")
print(df.head())

# Save the preprocessed data to a new CSV file
preprocessed_file_name = 'preprocessed_amazon_movie_ratings.csv'
df.to_csv(preprocessed_file_name, index=False)
print(f"\nPreprocessed data saved to '{preprocessed_file_name}'.")

# Zip the file
!zip preprocessed_data.zip {preprocessed_file_name}

# Download the zipped file
files.download('preprocessed_data.zip')
