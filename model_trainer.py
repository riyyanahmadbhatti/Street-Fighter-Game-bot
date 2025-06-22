import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(file_path):
    """
    Load and preprocess the dataset
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    print(f"Loading dataset from {file_path}...")
    
    # Load dataset
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    
    # Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values. Filling with 0.")
        df = df.fillna(0)
    
    # Display value counts for the target variable
    print("\nAction distribution:")
    print(df['action'].value_counts())
    
    # Handle potential imbalance
    min_samples = df['action'].value_counts().min()
    if min_samples < 10:
        print("\nWARNING: Some actions have very few samples.")
    
    # Split features and target
    X = df.drop('action', axis=1)
    y = df['action']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train the machine learning model
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    print("\nTraining model...")
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    # Parameter grid for GridSearchCV
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5, 10]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(
        pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    """
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    # Get feature importances (for RandomForest)
    try:
        # Extract the classifier from the pipeline
        classifier = model.named_steps['classifier']
        
        # Get feature importances
        importances = classifier.feature_importances_
        feature_names = X_test.columns
        
        # Sort feature importances
        indices = np.argsort(importances)[::-1]
        
        # Print top 10 features
        print("\nTop 10 features:")
        for i in range(min(10, len(feature_names))):
            print(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
        
        # Plot feature importances
        plt.figure(figsize=(12, 6))
        plt.title("Feature Importances")
        plt.bar(range(min(20, len(indices))), 
                importances[indices[:20]],
                align="center")
        plt.xticks(range(min(20, len(indices))), 
                  [feature_names[i] for i in indices[:20]], 
                  rotation=90)
        plt.tight_layout()
        plt.savefig("feature_importances.png")
        print("\nFeature importance plot saved as 'feature_importances.png'")
        
    except Exception as e:
        print(f"Could not extract feature importances: {e}")

def main():
    """Main function for model training"""
    # Path to dataset file
    dataset_file = "sf2_dataset.csv"
    
    if not os.path.exists(dataset_file):
        print(f"Dataset file {dataset_file} not found.")
        
        # Check if player-specific datasets exist
        p1_file = "sf2_dataset_p1.csv"
        p2_file = "sf2_dataset_p2.csv"
        
        if os.path.exists(p1_file) and os.path.exists(p2_file):
            print(f"Found player-specific datasets. Merging {p1_file} and {p2_file}...")
            
            # Load and merge datasets
            df1 = pd.read_csv(p1_file)
            df2 = pd.read_csv(p2_file)
            merged_df = pd.concat([df1, df2], ignore_index=True)
            
            # Save merged dataset
            merged_df.to_csv(dataset_file, index=False)
            print(f"Merged dataset saved to {dataset_file}")
        else:
            print("No dataset files found. Please run data collection first.")
            return
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test = load_and_preprocess_data(dataset_file)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save model
    model_file = "sf2_model.joblib"
    joblib.dump(model, model_file)
    print(f"\nModel saved to {model_file}")

if __name__ == "__main__":
    main()