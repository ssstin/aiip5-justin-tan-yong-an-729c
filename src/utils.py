# utils.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


def load_and_preprocess_data(df):
    """
    Preprocess the dataframe for model training
    Returns X_temp, y_temp, X_class, y_class and preprocessed dataframe
    """
    # Create Plant_Type_Stage column for classification
    conditions = [
        df['PlantType_Vine Crops_Stage'] > 0,
        df['PlantType_Herbs_Stage'] > 0,
        df['PlantType_Fruiting Vegetables_Stage'] > 0,
        df['PlantType_Leafy Greens_Stage'] > 0
    ]
    values = [
        'Vine_' + df['PlantType_Vine Crops_Stage'].astype(str),
        'Herbs_' + df['PlantType_Herbs_Stage'].astype(str),
        'Fruiting_' + df['PlantType_Fruiting Vegetables_Stage'].astype(str),
        'Leafy_' + df['PlantType_Leafy Greens_Stage'].astype(str)
    ]
    df['Plant_Type_Stage'] = np.select(conditions, values, default='Unknown')
    
    # Prepare data for temperature prediction
    X_temp = df.drop(['Temperature_Sensor_Clean', 'PlantType_Vine Crops_Stage',
                      'PlantType_Herbs_Stage', 'PlantType_Fruiting Vegetables_Stage',
                      'PlantType_Leafy Greens_Stage', 'Plant_Type_Stage', 'Cluster'], axis=1)
    y_temp = df['Temperature_Sensor_Clean']
    
    # Prepare data for plant type-stage classification
    X_class = df.drop(['PlantType_Vine Crops_Stage', 'PlantType_Herbs_Stage',
                      'PlantType_Fruiting Vegetables_Stage', 'PlantType_Leafy Greens_Stage',
                      'Plant_Type_Stage', 'Cluster'], axis=1)
    y_class = df['Plant_Type_Stage']
    
    return X_temp, y_temp, X_class, y_class, df


def scale_features(X_train, X_test):
    """Scale numerical features using StandardScaler"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames with column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler


def plot_feature_importance(feature_importances, title, top_n=20):
    """Plot feature importances from a trained model"""
    plt.figure(figsize=(12, 8))
    n_features = min(top_n, len(feature_importances))
    sns.barplot(
        x='importance',
        y='feature',
        data=feature_importances.head(n_features)
    )
    plt.title(f'Top {n_features} Feature Importances: {title}')
    plt.tight_layout()
    plt.show()


def plot_regression_results(y_true, y_pred, title):
    """Plot actual vs. predicted values for regression tasks"""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('Actual Temperature')
    plt.ylabel('Predicted Temperature')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(conf_matrix, classes, title):
    """Plot confusion matrix for classification tasks"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes
    )
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.show()