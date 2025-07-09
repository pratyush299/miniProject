import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from collections import Counter
import joblib

# Load data
data = pd.read_csv("Depression.csv")

# Drop rows where target is missing
data = data.dropna(subset=["Depression State"])

# Clean 'Depression State'
data["Depression State"] = (
    data["Depression State"].str.strip()
    .str.replace("\t", "")
    .str.replace(r"\d+", "", regex=True)
    .str.lower()
)

# Fill missing values in features
feature_columns = data.columns.difference(["Depression State", "Number"])
data[feature_columns] = data[feature_columns].fillna(data[feature_columns].median())

# Encode target
encoder = LabelEncoder()
data["Depression State"] = encoder.fit_transform(data["Depression State"])
print("Unique classes:", encoder.classes_)

# Features and target
X = data[feature_columns]
y = data["Depression State"]
print("Original class distribution:", Counter(y))

# Balance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("Class distribution after SMOTE:", Counter(y_resampled))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CatBoost parameter grid
param_grid = {
    'depth': [6, 8],
    'learning_rate': [0.01, 0.1],
    'iterations': [100, 200],
    'l2_leaf_reg': [3, 5]
}

cat = CatBoostClassifier(verbose=0, random_state=42)

grid = GridSearchCV(cat, param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1)
grid.fit(X_train_scaled, y_train)

best_cat = grid.best_estimator_
print("Best Params:", grid.best_params_)

# Predictions
y_pred = best_cat.predict(X_test_scaled)

# Class names
actual_class_names = [encoder.classes_[i] for i in sorted(set(np.unique(y_test)) | set(np.unique(y_pred)))]

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=actual_class_names))

# Save artifacts
joblib.dump(best_cat, "catboost_model.pkl")
joblib.dump(encoder, "label_encoder.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_columns.tolist(), "feature_columns.pkl")

print("✅ CatBoost model trained and saved!")
