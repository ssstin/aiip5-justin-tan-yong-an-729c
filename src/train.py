# train.py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import pandas as pd

from utils import (
    plot_feature_importance, 
    plot_regression_results, 
    plot_confusion_matrix
)
from model import (
    TemperaturePredictor, 
    RFTemperaturePredictor,
    PlantTypeStageClassifier, 
    SVMClassifier
)
from config import (
    TemperatureModel, ClassificationModel, Training
)


def setup_plotting():
    """Set up plotting style"""
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})


def train_temperature_models(X, y, hyperparameter_tuning=True, show_plots=False):
    """Train and evaluate multiple temperature prediction models"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Training.TEST_SIZE, random_state=42
    )
    
    # Initialize models
    xgb_model = TemperaturePredictor(**TemperatureModel.PARAMS)
    rf_model = RFTemperaturePredictor(**TemperatureModel.RF_PARAMS)
    
    # Train XGBoost model
    print("\n=== Training XGBoost for Temperature Prediction ===")
    if hyperparameter_tuning:
        xgb_model.tune_hyperparameters(
            X_train, y_train, 
            param_grid=TemperatureModel.PARAM_GRID,
            n_iter=Training.RANDOM_SEARCH_ITER,
            cv=Training.RANDOM_SEARCH_CV
        )
    else:
        print("Training with default parameters...")
        xgb_model.train(X_train, y_train)
    
    # Evaluate XGBoost model
    xgb_eval = xgb_model.evaluate(X_test, y_test)
    print(f"XGBoost - RMSE: {xgb_eval['rmse']:.4f}, R²: {xgb_eval['r2']:.4f}")
    
    # Cross-validation for XGBoost
    xgb_cv = xgb_model.cross_validate(X, y, cv=Training.CV_FOLDS)
    print(f"CV Results - RMSE: {xgb_cv['cv_rmse_mean']:.4f} ± {xgb_cv['cv_rmse_std']:.4f}, " 
          f"R²: {xgb_cv['cv_r2_mean']:.4f} ± {xgb_cv['cv_r2_std']:.4f}")
    
    # Plot XGBoost results if enabled
    if show_plots:
        plot_feature_importance(xgb_model.feature_importances, "XGBoost Temperature Prediction")
        plot_regression_results(y_test, xgb_eval['predictions'], "XGBoost Temperature Prediction Results")
    
    # Train Random Forest model
    print("\n=== Training Random Forest for Temperature Prediction ===")
    if hyperparameter_tuning:
        rf_model.tune_hyperparameters(
            X_train, y_train, 
            param_grid=TemperatureModel.RF_PARAM_GRID,
            n_iter=Training.RANDOM_SEARCH_ITER,
            cv=Training.RANDOM_SEARCH_CV
        )
    else:
        print("Training with default parameters...")
        rf_model.train(X_train, y_train)
    
    # Evaluate Random Forest model
    rf_eval = rf_model.evaluate(X_test, y_test)
    print(f"Random Forest - RMSE: {rf_eval['rmse']:.4f}, R²: {rf_eval['r2']:.4f}")
    
    # Cross-validation for Random Forest
    rf_cv = rf_model.cross_validate(X, y, cv=Training.CV_FOLDS)
    print(f"CV Results - RMSE: {rf_cv['cv_rmse_mean']:.4f} ± {rf_cv['cv_rmse_std']:.4f}, " 
          f"R²: {rf_cv['cv_r2_mean']:.4f} ± {rf_cv['cv_r2_std']:.4f}")
    
    # Plot Random Forest results if enabled
    if show_plots:
        plot_feature_importance(rf_model.feature_importances, "Random Forest Temperature Prediction")
        plot_regression_results(y_test, rf_eval['predictions'], "Random Forest Temperature Prediction Results")
    
    # Save RF model (optional)
    rf_model.save(TemperatureModel.RF_MODEL_PATH)
    
    # Return both models for comparison
    return [xgb_model, rf_model]


def train_classification_models(X, y, hyperparameter_tuning=True, show_plots=False):
    """Train and evaluate multiple plant type-stage classification models"""
    # Split data for classification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Training.TEST_SIZE, random_state=42, stratify=y
    )
    
    # Initialize models
    rf_model = PlantTypeStageClassifier(**ClassificationModel.PARAMS)
    svm_model = SVMClassifier(**ClassificationModel.SVM_PARAMS)
    
    # Train Random Forest model
    print("\n=== Training Random Forest for Classification ===")
    if hyperparameter_tuning:
        rf_model.tune_hyperparameters(
            X_train, y_train, 
            param_grid=ClassificationModel.PARAM_GRID,
            n_iter=Training.RANDOM_SEARCH_ITER,
            cv=Training.RANDOM_SEARCH_CV
        )
    else:
        print("Training with default parameters...")
        rf_model.train(X_train, y_train)
    
    # Evaluate Random Forest model
    rf_eval = rf_model.evaluate(X_test, y_test)
    print(f"Random Forest Classification - Accuracy: {rf_eval['accuracy']:.4f}")
    print("Classification Report:")
    for cls, metrics in rf_eval['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"  {cls}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    # Cross-validation for Random Forest
    rf_cv = rf_model.cross_validate(X, y, cv=Training.CV_FOLDS)
    print(f"CV Accuracy: {rf_cv['cv_accuracy_mean']:.4f} ± {rf_cv['cv_accuracy_std']:.4f}")
    
    # Plot Random Forest results if enabled
    if show_plots:
        if hasattr(rf_model, 'feature_importances'):
            plot_feature_importance(rf_model.feature_importances, "Random Forest Classification")
        plot_confusion_matrix(
            rf_eval['confusion_matrix'], 
            rf_model.classes, 
            "Random Forest Classification Results"
        )
    
    # Train SVM model
    print("\n=== Training SVM for Classification ===")
    if hyperparameter_tuning:
        svm_model.tune_hyperparameters(
            X_train, y_train, 
            param_grid=ClassificationModel.SVM_PARAM_GRID,
            n_iter=Training.RANDOM_SEARCH_ITER,
            cv=Training.RANDOM_SEARCH_CV
        )
    else:
        print("Training with default parameters...")
        svm_model.train(X_train, y_train)
    
    # Evaluate SVM model
    svm_eval = svm_model.evaluate(X_test, y_test)
    print(f"SVM Classification - Accuracy: {svm_eval['accuracy']:.4f}")
    print("Classification Report:")
    for cls, metrics in svm_eval['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"  {cls}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    # Cross-validation for SVM
    svm_cv = svm_model.cross_validate(X, y, cv=Training.CV_FOLDS)
    print(f"CV Accuracy: {svm_cv['cv_accuracy_mean']:.4f} ± {svm_cv['cv_accuracy_std']:.4f}")
    
    # Plot SVM results if enabled
    if show_plots:
        plot_confusion_matrix(
            svm_eval['confusion_matrix'], 
            svm_model.classes, 
            "SVM Classification Results"
        )
    
    # Save SVM model (optional)
    svm_model.save(ClassificationModel.SVM_MODEL_PATH)
    
    # Return both models for comparison
    return [rf_model, svm_model]