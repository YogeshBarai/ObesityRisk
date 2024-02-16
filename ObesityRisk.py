import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the training and test datasets
train_data = pd.read_csv("./input/train.csv")
test_data = pd.read_csv("./input/test.csv")

# Encode the target variable 'NObeyesdad' in the training dataset
label_encoder = LabelEncoder()
train_data['NObeyesdad_encoded'] = label_encoder.fit_transform(train_data['NObeyesdad'])

# Separate features (X) and target variable (y) in the training dataset
X = train_data.drop(['id', 'NObeyesdad', 'NObeyesdad_encoded'], axis=1)
y = train_data['NObeyesdad_encoded']

# One-hot encode categorical variables in the training dataset
X = pd.get_dummies(X)

# Align the columns of the test dataset with the training dataset
test_data_processed = pd.get_dummies(test_data.drop('id', axis=1))
test_data_processed = test_data_processed.reindex(columns=X.columns, fill_value=0)

# Split the training dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)

# Train the Random Forest Classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict the target variable on the validation set
y_pred = rf_classifier.predict(X_val)

# Calculate accuracy on the validation set
accuracy = accuracy_score(y_val, y_pred)
print("\nAccuracy:", accuracy)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred))

# Predict the target variable on the test dataset
test_predictions = rf_classifier.predict(test_data_processed)

# Decode the encoded predictions back to original labels
test_predictions_decoded = label_encoder.inverse_transform(test_predictions)

# Create a DataFrame for test predictions
test_predictions_df = pd.DataFrame({'id': test_data['id'], 'NObeyesdad': test_predictions_decoded})

# Save the predictions to a CSV file
test_predictions_df.to_csv("predictions.csv", index=False)
