# model.py
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, KFold, StratifiedKFold, RandomizedSearchCV


class TemperaturePredictor:
    """Model for predicting temperature based on environmental conditions using XGBoost"""
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1):
        """Initialize the temperature prediction model"""
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.feature_importances = None
        self.best_params = None
    
    def tune_hyperparameters(self, X, y, param_grid, n_iter=10, cv=3, verbose=1):
        """Tune hyperparameters using RandomizedSearchCV"""
        print("Tuning hyperparameters for temperature model...")
        random_search = RandomizedSearchCV(
            estimator=XGBRegressor(random_state=42, n_jobs=-1),
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            verbose=verbose,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X, y)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best RMSE: {-random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        return self
        
    def train(self, X_train, y_train):
        """Train the model using provided training data"""
        self.model.fit(X_train, y_train)
        self.feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return self
        
    def predict(self, X):
        """Generate temperature predictions for input features"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test data"""
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        return {
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions
        }
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation to assess model robustness"""
        cv_rmse = -cross_val_score(
            self.model, X, y, 
            scoring='neg_root_mean_squared_error', 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42)
        )
        cv_r2 = cross_val_score(
            self.model, X, y, 
            scoring='r2', 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42)
        )
        return {
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }
    
    def save(self, filepath):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        return self


class RFTemperaturePredictor:
    """Model for predicting temperature based on environmental conditions using Random Forest"""
    
    def __init__(self, n_estimators=100, max_depth=None, random_state=42, n_jobs=-1):
        """Initialize the temperature prediction model"""
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=n_jobs
        )
        self.feature_importances = None
        self.best_params = None
    
    def tune_hyperparameters(self, X, y, param_grid, n_iter=10, cv=3, verbose=1):
        """Tune hyperparameters using RandomizedSearchCV"""
        print("Tuning hyperparameters for Random Forest temperature model...")
        random_search = RandomizedSearchCV(
            estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='neg_root_mean_squared_error',
            cv=cv,
            verbose=verbose,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X, y)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best RMSE: {-random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        return self
        
    def train(self, X_train, y_train):
        """Train the model using provided training data"""
        self.model.fit(X_train, y_train)
        self.feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return self
        
    def predict(self, X):
        """Generate temperature predictions for input features"""
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test data"""
        predictions = self.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)
        return {
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions
        }
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation to assess model robustness"""
        cv_rmse = -cross_val_score(
            self.model, X, y, 
            scoring='neg_root_mean_squared_error', 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42)
        )
        cv_r2 = cross_val_score(
            self.model, X, y, 
            scoring='r2', 
            cv=KFold(n_splits=cv, shuffle=True, random_state=42)
        )
        return {
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }
    
    def save(self, filepath):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        return self


class PlantTypeStageClassifier:
    """Model for classifying plant type and growth stage"""
    
    def __init__(self, n_estimators=100, random_state=42, n_jobs=-1, **kwargs):
        """Initialize the plant type-stage classification model"""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs,
            class_weight='balanced', 
            **kwargs
        )
        self.feature_importances = None
        self.classes = None
        self.best_params = None
        
    def tune_hyperparameters(self, X, y, param_grid, n_iter=10, cv=3, verbose=1):
        """Tune hyperparameters using RandomizedSearchCV"""
        print("Tuning hyperparameters for classification model...")
        random_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X, y)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best accuracy: {random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.classes = random_search.best_estimator_.classes_
        
        return self
        
    def train(self, X_train, y_train):
        """Train the model using provided training data"""
        self.model.fit(X_train, y_train)
        self.classes = self.model.classes_
        self.feature_importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        return self
        
    def predict(self, X):
        """Generate plant type-stage predictions for input features"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Generate probability estimates for each class"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test data"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions
        }
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation to assess model robustness"""
        cv_accuracy = cross_val_score(
            self.model, X, y, 
            scoring='accuracy', 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        )
        cv_predictions = cross_val_predict(self.model, X, y, cv=cv)
        cv_report = classification_report(y, cv_predictions, output_dict=True)
        
        return {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_report': cv_report
        }
    
    def save(self, filepath):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        self.classes = self.model.classes_
        return self


class SVMClassifier:
    """Model for classifying plant type and growth stage using SVM"""
    
    def __init__(self, C=1.0, kernel='rbf', random_state=42, **kwargs):
        """Initialize the plant type-stage classification model"""
        self.model = SVC(
            C=C,
            kernel=kernel,
            random_state=random_state,
            probability=True,
            class_weight='balanced',
            **kwargs
        )
        self.classes = None
        self.best_params = None
        
    def tune_hyperparameters(self, X, y, param_grid, n_iter=10, cv=3, verbose=1):
        """Tune hyperparameters using RandomizedSearchCV"""
        print("Tuning hyperparameters for SVM classification model...")
        random_search = RandomizedSearchCV(
            estimator=SVC(random_state=42, probability=True, class_weight='balanced'),
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring='accuracy',
            cv=cv,
            verbose=verbose,
            random_state=42,
            n_jobs=-1
        )
        random_search.fit(X, y)
        
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best accuracy: {random_search.best_score_:.4f}")
        
        # Update model with best parameters
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        self.classes = random_search.best_estimator_.classes_
        
        return self
        
    def train(self, X_train, y_train):
        """Train the model using provided training data"""
        self.model.fit(X_train, y_train)
        self.classes = self.model.classes_
        return self
        
    def predict(self, X):
        """Generate plant type-stage predictions for input features"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Generate probability estimates for each class"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance on test data"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        conf_matrix = confusion_matrix(y_test, predictions)
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': conf_matrix,
            'predictions': predictions
        }
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation to assess model robustness"""
        cv_accuracy = cross_val_score(
            self.model, X, y, 
            scoring='accuracy', 
            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        )
        cv_predictions = cross_val_predict(self.model, X, y, cv=cv)
        cv_report = classification_report(y, cv_predictions, output_dict=True)
        
        return {
            'cv_accuracy_mean': cv_accuracy.mean(),
            'cv_accuracy_std': cv_accuracy.std(),
            'cv_report': cv_report
        }
    
    def save(self, filepath):
        """Save model to disk"""
        joblib.dump(self.model, filepath)
        
    def load(self, filepath):
        """Load model from disk"""
        self.model = joblib.load(filepath)
        self.classes = self.model.classes_
        return self