import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %%
# Feature columns
FEATURES = [
    'property_type',
    'proximity_to_food',
    'proximity_to_education',
    'proximity_to_culture',
    'proximity_to_central_london',
    'ptal',
    'lagged_crime_rate',
    'lagged_cpih_inflation',
    'lagged_mortgage_rates',
    'lagged_population_growth',
    'lagged_household_income',
    'lagged_unemployment_rate',
    'lagged_price_per_square_meter'
]

TARGET = 'price'  # Adjust based on your target column name

# %%
def load_data(data_path):
    """
    Load and prepare data for training.
    
    Args:
        data_path: Path to the data file
        
    Returns:
        X: Feature matrix
        y: Target vector
    """
    df = pd.read_csv(data_path)
    
    # Select features and target
    X = df[FEATURES].copy()
    y = df[TARGET].copy()
    
    # Handle categorical features if needed
    if 'property_type' in X.columns:
        X['property_type'] = X['property_type'].astype('category')
    
    return X, y

# %%
def objective(trial, X_train, y_train, X_val, y_val):
    """
    Optuna objective function for hyperparameter tuning.
    
    Args:
        trial: Optuna trial object
        X_train: Training features
        y_train: Training target
        X_val: Validation features
        y_val: Validation target
        
    Returns:
        Validation RMSE
    """
    # Define hyperparameter search space
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'num_leaves': trial.suggest_int('num_leaves', 10, 300),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'verbosity': -1,
        'random_state': 42
    }
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    # Make predictions and calculate RMSE
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    
    return rmse


def train_model(X, y, n_trials=50):
    """
    Train LightGBM model with Optuna hyperparameter tuning.
    
    Args:
        X: Feature matrix
        y: Target vector
        n_trials: Number of Optuna trials
        
    Returns:
        best_model: Trained LightGBM model
        best_params: Best hyperparameters
        study: Optuna study object
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Create Optuna study
    study = optuna.create_study(direction='minimize')
    
    # Optimize hyperparameters
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=n_trials,
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params
    best_params.update({
        'objective': 'regression',
        'metric': 'rmse',
        'boosting_type': 'gbdt',
        'verbosity': -1,
        'random_state': 42
    })
    
    # Train final model with best parameters
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    best_model = lgb.train(
        best_params,
        train_data,
        valid_sets=[val_data],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    return best_model, best_params, study


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance.
    
    Args:
        model: Trained LightGBM model
        X_test: Test features
        y_test: Test target
        
    Returns:
        Dictionary with evaluation metrics
    """
    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }
    
    return metrics


if __name__ == '__main__':
    # Example usage
    # data_path = 'path/to/your/data.csv'
    # X, y = load_data(data_path)
    # model, best_params, study = train_model(X, y, n_trials=50)
    # 
    # # Evaluate on test set
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # metrics = evaluate_model(model, X_test, y_test)
    # print(f"Best parameters: {best_params}")
    # print(f"Test metrics: {metrics}")
    pass


# %%
