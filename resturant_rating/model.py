import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv(r'C:\Users\abhin\Desktop\resturant_rating\Dataset .csv')

# Drop unnecessary columns
drop_cols = ["Restaurant ID", "Locality", "Locality Verbose", 
             "Is delivering now", "Switch to order menu", 
             "Restaurant Name", "City", "Address", "Currency","Longitude","Latitude","Country Code"]
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Handle missing values
df.dropna(inplace=True)

# Define categorical columns for Label Encoding
categorical_columns = ["Has Table booking", "Has Online delivery","Cuisines"]

# Encode categorical features
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le  # Store encoder for future use

# Function to remove outliers using IQR method
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df[column] = np.where(df[column] > upper_bound, upper_bound,
                          np.where(df[column] < lower_bound, lower_bound, df[column]))
    return df

# Apply outlier removal to specific columns
outlier_columns = ["Votes", "Average Cost for two", "Price range", "Aggregate rating"]
for col in outlier_columns:
    df = remove_outliers_iqr(df, col)

# Visualize the impact of outlier removal using KDE plots
plt.figure(figsize=(12, 8))
for col in outlier_columns:
    sns.kdeplot(df[col], label=col)
plt.legend()
plt.title("Distribution of Features After Outlier Removal")
plt.show()

# Convert Aggregate Rating into Categorical Classes for Classification
def categorize_rating(rating):
    if rating <= 2.5:
        return 0  # Low
    elif 2.5 < rating <= 3.5:
        return 1  # Medium
    else:
        return 2  # High

df["Rating Category"] = df["Aggregate rating"].apply(categorize_rating)

# Define features (X) and target (y)
x = df.drop(columns=["Aggregate rating", "Rating Category","Rating color","Rating text"], errors='ignore')  # Features
y = df["Rating Category"]  # Target variable

print(x)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42, stratify=y)

# Define Decision Tree Classifier
tree = DecisionTreeClassifier(random_state=42)

# Define parameter grid for hyperparameter tuning
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [2, 4, 5, 6, 10, 20,50,100],
    'max_features': [None, 'sqrt', 'log2','auto'],
    'min_samples_split': [10, 20, 30, 50,60,70,100],
    'class_weight': [{0: 1, 1: 1, 2: 2}, {0: 1, 1: 2, 2: 1}],
    'ccp_alpha': [0.0001, 0.001, 0.01, 0.1, 0.2],
    'min_impurity_decrease': [0.0, 0.01, 0.05, 0.1]
}

# Stratified K-Fold for better cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Perform Grid Search
grid_tree = GridSearchCV(tree, param_grid=param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid_tree.fit(X_train, y_train)

# Save the best model & encoders
# joblib.dump(grid_tree.best_estimator_, 'trained_restaurant_rating_classifier.pkl')
# joblib.dump(label_encoders, 'label_encoders.pkl')

# Print best parameters & accuracy
print("\n Best Parameters:", grid_tree.best_params_)
print(" Best Cross-Validation Score:", round(grid_tree.best_score_, 4))

# Make predictions
y_pred = grid_tree.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_pred,y_test)
print("\n Accuracy Score:", round(accuracy, 4))
print("\n Classification Report:\n", classification_report(y_pred,y_test))
