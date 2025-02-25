import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
data = pd.read_csv("Depression.csv")

# Drop rows where 'Depression State' is missing
data = data.dropna(subset=["Depression State"])

# Clean the 'Depression State' column
data["Depression State"] = data["Depression State"].str.strip()  # Remove leading/trailing spaces
data["Depression State"] = data["Depression State"].str.replace("\t", "")  # Remove tabs
data["Depression State"] = data["Depression State"].str.replace(r"\d+", "", regex=True)# Remove numbers
data["Depression State"] = data["Depression State"].str.lower()  

print(data["Depression State"].unique())

# Fill missing values in feature columns with the median
feature_columns = data.columns.difference(["Depression State", "Number"])
data[feature_columns] = data[feature_columns].fillna(data[feature_columns].median())

# Encode the target column
encoder = LabelEncoder()
data["Depression State"] = encoder.fit_transform(data["Depression State"])

# Define features and target
X = data[feature_columns]
y = data["Depression State"]

# Print class distribution
print("Original class distribution:", Counter(y))

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", Counter(y_resampled))

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Get the unique classes that actually appear in the test set
unique_classes = sorted(list(set(y_test) | set(y_pred)))
actual_class_names = [encoder.classes_[i] for i in unique_classes]

# Print classification report
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=actual_class_names))

# Save the model, encoder, and feature columns
joblib.dump(model, "depression_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")

print("Model trained and saved!")