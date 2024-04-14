import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset from CSV file
df = pd.read_csv('flood_prediction_data.csv')

# Features and target variable
X = df[['water_level_distance', 'water_opacity', 'expected_precipitation']]
y = df['flood']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy}")

# Save the trained model
with open('flood_model.pkl', 'wb') as f:
    pickle.dump(model, f)
