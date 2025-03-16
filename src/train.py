# train.py
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from utils import (
    plot_feature_importance, 
    plot_regression_results, 
    plot_confusion_matrix
)
from model import TemperaturePredictor, PlantTypeStageClassifier
from config import (
    TemperatureModel, ClassificationModel, Training
)


def setup_plotting():
    """Set up plotting style"""
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})


def train_temperature_model(X, y, hyperparameter_tuning=True, show_plots=False):
    """Train and evaluate temperature prediction model"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Training.TEST_SIZE, random_state=42
    )
    
    # Initialize temperature model
    temp_model = TemperaturePredictor(**TemperatureModel.PARAMS)
    
    # Hyperparameter tuning if requested
    if hyperparameter_tuning:
        temp_model.tune_hyperparameters(
            X_train, y_train, 
            param_grid=TemperatureModel.PARAM_GRID,
            n_iter=Training.RANDOM_SEARCH_ITER,
            cv=Training.RANDOM_SEARCH_CV
        )
    else:
        # Train model with default parameters
        print("Training temperature prediction model...")
        temp_model.train(X_train, y_train)
    
    # Evaluate model
    temp_eval = temp_model.evaluate(X_test, y_test)
    print(f"Temperature Model - RMSE: {temp_eval['rmse']:.4f}, R²: {temp_eval['r2']:.4f}")
    
    # Cross-validation
    temp_cv = temp_model.cross_validate(X, y, cv=Training.CV_FOLDS)
    print(f"CV Results - RMSE: {temp_cv['cv_rmse_mean']:.4f} ± {temp_cv['cv_rmse_std']:.4f}, " 
          f"R²: {temp_cv['cv_r2_mean']:.4f} ± {temp_cv['cv_r2_std']:.4f}")
    
    # Plot results if enabled
    if show_plots:
        plot_feature_importance(temp_model.feature_importances, "Temperature Prediction")
        plot_regression_results(y_test, temp_eval['predictions'], "Temperature Prediction Results")
    
    return temp_model


def train_classification_model(X, y, hyperparameter_tuning=True, show_plots=False):
    """Train and evaluate plant type-stage classification model"""
    # Split data for classification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Training.TEST_SIZE, random_state=42, stratify=y
    )
    
    # Initialize classification model
    class_model = PlantTypeStageClassifier(**ClassificationModel.PARAMS)
    
    # Hyperparameter tuning if requested
    if hyperparameter_tuning:
        class_model.tune_hyperparameters(
            X_train, y_train, 
            param_grid=ClassificationModel.PARAM_GRID,
            n_iter=Training.RANDOM_SEARCH_ITER,
            cv=Training.RANDOM_SEARCH_CV
        )
    else:
        # Train model with default parameters
        print("\nTraining plant type-stage classification model...")
        class_model.train(X_train, y_train)
    
    # Evaluate model
    class_eval = class_model.evaluate(X_test, y_test)
    print(f"Classification Model - Accuracy: {class_eval['accuracy']:.4f}")
    print("Classification Report:")
    for cls, metrics in class_eval['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"  {cls}: Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, F1={metrics['f1-score']:.4f}")
    
    # Cross-validation
    class_cv = class_model.cross_validate(X, y, cv=Training.CV_FOLDS)
    print(f"CV Accuracy: {class_cv['cv_accuracy_mean']:.4f} ± {class_cv['cv_accuracy_std']:.4f}")
    
    # Plot results if enabled
    if show_plots:
        plot_feature_importance(class_model.feature_importances, "Plant Type-Stage Classification")
        plot_confusion_matrix(
            class_eval['confusion_matrix'], 
            class_model.classes, 
            "Plant Type-Stage Classification Results"
        )
    
    return class_model