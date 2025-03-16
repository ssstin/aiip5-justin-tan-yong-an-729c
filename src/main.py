# main.py
import os
import sqlite3
import pandas as pd

from utils import load_and_preprocess_data
from train import train_temperature_model, train_classification_model, setup_plotting
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
    
    # Train temperature model
    temp_model = train_temperature_model(
        X_temp, y_temp,
        hyperparameter_tuning=Training.ENABLE_HYPERPARAMETER_TUNING,
        show_plots=Training.SHOW_PLOTS
    )
    
    # Train classification model
    class_model = train_classification_model(
        X_class, y_class,
        hyperparameter_tuning=Training.ENABLE_HYPERPARAMETER_TUNING,
        show_plots=Training.SHOW_PLOTS
    )
    
    # Save models
    print(f"\nSaving temperature model to {TemperatureModel.MODEL_PATH}...")
    temp_model.save(TemperatureModel.MODEL_PATH)
    
    print(f"Saving classification model to {ClassificationModel.MODEL_PATH}...")
    class_model.save(ClassificationModel.MODEL_PATH)
    
    print("\nModels trained and saved successfully.")


if __name__ == "__main__":
    main()