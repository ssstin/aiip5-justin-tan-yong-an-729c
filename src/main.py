# main.py
import os
import sqlite3
import pandas as pd

from utils import load_and_preprocess_data
from train import (
    train_temperature_models,
    train_classification_models,
    setup_plotting
)
from config import Database, TemperatureModel, ClassificationModel, Training


def load_data():
    """Load data from SQLite database"""
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(Database.PATH), exist_ok=True)
    
    # Connect to database and load data
    conn = sqlite3.connect(Database.PATH)
    df = pd.read_sql_query(Database.QUERY, conn)
    conn.close()
    
    return df

def main():
    """Main function to run the ML pipeline"""
    # Set up plotting if needed
    if Training.SHOW_PLOTS:
        setup_plotting()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    df = load_data()
    X_temp, y_temp, X_class, y_class, preprocessed_df = load_and_preprocess_data(df)
    
    # Filter features
    X_temp = preprocessed_df[TemperatureModel.FEATURES]
    X_class = preprocessed_df[ClassificationModel.FEATURES]
    
    # Train and compare temperature prediction models
    temp_models = train_temperature_models(
        X_temp, y_temp,
        hyperparameter_tuning=Training.ENABLE_HYPERPARAMETER_TUNING,
        show_plots=Training.SHOW_PLOTS
    )
    
    # Train and compare classification models
    class_models = train_classification_models(
        X_class, y_class,
        hyperparameter_tuning=Training.ENABLE_HYPERPARAMETER_TUNING,
        show_plots=Training.SHOW_PLOTS
    )
    
    # Print comparison results manually
    print("\n=== Model Comparison Results ===")
    
    print("\nTemperature Model Comparison:")
    print("XGBoost vs RandomForest - See training output above for metrics")
    
    print("\nClassification Model Comparison:")
    print("RandomForest vs SVM - See training output above for metrics")
    
    # Get best models based on performance
    best_temp_model = temp_models[0]  
    best_class_model = class_models[0]  
    
    # Save best models
    print(f"\nSaving best temperature model to {TemperatureModel.MODEL_PATH}...")
    best_temp_model.save(TemperatureModel.MODEL_PATH)
    
    print(f"Saving best classification model to {ClassificationModel.MODEL_PATH}...")
    best_class_model.save(ClassificationModel.MODEL_PATH)
    
    print("\nModels trained and saved successfully.")


if __name__ == "__main__":
    main()