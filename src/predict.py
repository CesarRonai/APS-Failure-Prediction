import joblib
import pandas as pd

# Load trained model
model = joblib.load("../models/modelo_random_forest.pkl")

# Load new input data (substitua pelo seu arquivo de entrada)
X_new = pd.read_csv("../data/sample_input.csv")  # Exemplo de entrada

# Make predictions
predictions = model.predict(X_new)

# Display results
print("Predictions:", predictions)
