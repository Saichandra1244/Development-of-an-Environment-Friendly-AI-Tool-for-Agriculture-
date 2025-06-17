import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

import os  # Add this at the top of your file

def train_and_save_model():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Rest of your existing code...
    data = pd.read_csv("data/Crop_recommendation.csv")
    # ... continue with the rest of your function

def train_and_save_model():
    # Load data
    data = pd.read_csv("data/Crop_recommendation.csv")
    
    # Preprocess
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model with updated RandomForest parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        max_features='sqrt',
        random_state=42,
        verbose=1,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, "models/crop_recommender.pkl")
    print("Model saved successfully!")
    
    # Return feature names for reference
    return list(X.columns)

if __name__ == "__main__":
    train_and_save_model()
