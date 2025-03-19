import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv("../data/train_balanced.csv")  # Certifique-se de salvar os dados antes

# Separate features and target
X = data.drop(columns=["class"])
y = data["class"]

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Evaluate model
y_pred = rf.predict(X_test)
print("Model Performance:\n", classification_report(y_test, y_pred))

# Save trained model
joblib.dump(rf, "../models/modelo_random_forest.pkl")
print("Model saved successfully!")
