# -------------------------------------------------------------------------
# AUTHOR: Jason Mar
# FILENAME: pca.py
# SPECIFICATION: Short program to find the best column to drop to reduce
# dimensions for heart disease csv file
# FOR: CS 4440 (Data Mining) - Assignment #2
# TIME SPENT: 1 hour
# -----------------------------------------------------------*/

#importing some Python libraries
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Load the data
#--> add your Python code here
df = pd.read_csv('heart_disease_dataset.csv')


# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

#Get the number of features
#--> add your Python code here
num_features = scaled_data.shape[1]

# dictionary to hold the column we dropped and the variance we calculated
feature_removed_value = {}

# Run PCA using 9 features, removing one feature at each iteration
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    # --> add your Python code here

    # returns a copy array, so no need to re add the column we dropped each time
    reduced_data = np.delete(scaled_data, i, axis = 1)
    df_features = pd.DataFrame(reduced_data)

    # Run PCA on the reduced dataset
    # --> add your Python code here
    pca = PCA()
    X_pca = pca.fit_transform(df_features)

    #Store PC1 variance and the feature removed
    #Use pca.explained_variance_ratio_[0] and df_features.columns[i] for that
    # --> add your Python code here
    feature_removed_value[df.columns[i]] =  pca.explained_variance_ratio_[0]

# Find the maximum PC1 variance
# --> add your Python code here
maxPC1 = float('-inf')
feature_to_remove = ''
for key,value in feature_removed_value.items():
    if value > maxPC1:
        maxPC1 = value
        feature_to_remove = key

#Print results
#Use the format: Highest PC1 variance found: ? when removing ?
print(f"Highest PC1 variance found: {maxPC1} when removing {feature_to_remove}")





