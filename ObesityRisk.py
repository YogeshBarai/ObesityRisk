import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the training and test datasets
train_data = pd.read_csv("input/train.csv")
test_data = pd.read_csv("input/test.csv")

# Encode the target variable 'NObeyesdad' in the training dataset
label_encoder = LabelEncoder()
train_data['NObeyesdad_encoded'] = label_encoder.fit_transform(train_data['NObeyesdad'])

# Drop unnecessary columns and split features and target variable
X_train = train_data.drop(['id', 'NObeyesdad', 'NObeyesdad_encoded'], axis=1)
y_train = train_data['NObeyesdad_encoded']

# One-hot encode categorical variables in the training dataset
X_train = pd.get_dummies(X_train)

# Split the training dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform the validation data
X_val_scaled = scaler.transform(X_val)

# Initialize the Gradient Boosting Classifier
gb_classifier = GradientBoostingClassifier(random_state=42)

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}

# Perform Grid Search Cross Validation
grid_search = GridSearchCV(gb_classifier, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

# Get the best model from Grid Search
best_gb_classifier = grid_search.best_estimator_

# Predict the target variable on the scaled validation set
y_pred_val = best_gb_classifier.predict(X_val_scaled)

# Calculate accuracy on the validation set
accuracy_val = accuracy_score(y_val, y_pred_val)
print("Validation Accuracy:", accuracy_val)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val))

# Preprocess the test dataset
test_data_processed = pd.get_dummies(test_data.drop('id', axis=1))
test_data_processed = test_data_processed.reindex(columns=X_train.columns, fill_value=0)
test_data_scaled = scaler.transform(test_data_processed)

# Predict the target variable on the scaled test dataset
test_predictions = best_gb_classifier.predict(test_data_scaled)

# Decode the encoded predictions back to original labels
test_predictions_decoded = label_encoder.inverse_transform(test_predictions)

# Create a DataFrame for test predictions
test_predictions_df = pd.DataFrame({'id': test_data['id'], 'NObeyesdad': test_predictions_decoded})

# Save the predictions to a CSV file
test_predictions_df.to_csv("submission_3.csv", index=False)
