# config.py
"""Configuration settings for the machine learning pipeline"""

class Database:
    PATH = 'data/agri.db'
    QUERY = "SELECT * FROM farm_data"


class TemperatureModel:
    # Features for temperature prediction
    FEATURES = [
        'Humidity Sensor (%)', 
        'CO2 Sensor (ppm)', 
        'EC Sensor (dS/m)', 
        'O2 Sensor (ppm)',
        'Nutrient N Sensor (ppm)', 
        'Nutrient P Sensor (ppm)', 
        'Nutrient K Sensor (ppm)',
        'pH Sensor', 
        'Water Level Sensor (mm)', 
        'PrevPlant_Fruiting Vegetables',
        'PrevPlant_Vine Crops', 
        'Light_Intensity_Clean', 
        'Cluster_2'
    ]

    # Default hyperparameters
    PARAMS = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'random_state': 42
    }

    # Hyperparameter search space
    PARAM_GRID = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 6, 9]
    }

    # Model file path
    MODEL_PATH = 'src/models/temperature_model.joblib'


class ClassificationModel:
    # Features for plant type-stage classification
    FEATURES = [
        'Humidity Sensor (%)', 
        'CO2 Sensor (ppm)', 
        'EC Sensor (dS/m)', 
        'O2 Sensor (ppm)',
        'Nutrient N Sensor (ppm)', 
        'Nutrient P Sensor (ppm)', 
        'Nutrient K Sensor (ppm)',
        'pH Sensor', 
        'Water Level Sensor (mm)', 
        'Temperature_Sensor_Clean',
        'Light_Intensity_Clean', 
        'Cluster_0', 
        'Cluster_1'
    ]

    # Default hyperparameters
    PARAMS = {
        'n_estimators': 100,
        'random_state': 42
    }

    # Hyperparameter search space
    PARAM_GRID = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'class_weight': ['balanced', 'balanced_subsample']
    }

    # Model file path
    MODEL_PATH = 'src/models/plant_typestage_model.joblib'


class Training:
    # Cross-validation settings
    CV_FOLDS = 5
    TEST_SIZE = 0.2

    # Random search settings 
    RANDOM_SEARCH_ITER = 10
    RANDOM_SEARCH_CV = 3

    # Enable/disable hyperparameter tuning and plotting
    ENABLE_HYPERPARAMETER_TUNING = True
    SHOW_PLOTS = False