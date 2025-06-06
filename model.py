# model.py
import os
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib
import numpy as np

import optuna

import config

class StatPredictor(nn.Module):
    def __init__(self):
        super(StatPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(config.INPUT_SIZE, config.HIDDEN_LAYER_1),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_LAYER_1, config.HIDDEN_LAYER_2),
            nn.ReLU(),
            nn.Linear(config.HIDDEN_LAYER_2, config.OUTPUT_SIZE)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

def train_model(characters_df, karts_df, mario_combos_df, force_retrain=False):
    if os.path.exists(config.MODEL_PATH) and not force_retrain:
        print("\nFound existing model. Skipping training.")
        return

    print("\n--- Starting Model Training (Data Augmentation approach) ---")
    
    common_karts = karts_df.index.intersection(mario_combos_df.index)
    if len(common_karts) == 0:
        print("Error: No common karts found.")
        exit()
    print(f"Found {len(common_karts)} common karts. Augmenting data...")
        
    mario_stats = characters_df.loc['Mario'][['Speed', 'Acceleration', 'Weight', 'Handling']].values
    
    X_list, y_list = [], []
    ref_characters = ['Mario', 'Bowser', 'Baby Peach']
    
    for kart_name in common_karts:
        kart_stats = karts_df.loc[kart_name].values
        
        for char_name in ref_characters:
            char_stats = characters_df.loc[char_name][['Speed', 'Acceleration', 'Weight', 'Handling']].values
            base_delta = char_stats - mario_stats
            input_features = list(kart_stats) + list(base_delta)
            X_list.append(input_features)
            
            simulated_final_delta = base_delta * config.STAT_FACTOR
            y_list.append(simulated_final_delta)

    X = torch.tensor(np.array(X_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    
    print(f"Augmented training data created with {len(X)} samples.")
    
    scaler_X = StandardScaler().fit(X)
    X_scaled = torch.tensor(scaler_X.transform(X), dtype=torch.float32)
    scaler_y = StandardScaler().fit(y)
    y_scaled = torch.tensor(scaler_y.transform(y), dtype=torch.float32)

    model = StatPredictor()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    for epoch in range(config.EPOCHS):
        model.train()
        outputs = model(X_scaled)
        loss = criterion(outputs, y_scaled)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 400 == 0:
            print(f'Epoch [{epoch+1}/{config.EPOCHS}], Loss: {loss.item():.8f}')

    torch.save(model.state_dict(), config.MODEL_PATH)
    joblib.dump(scaler_X, config.SCALER_X_PATH)
    joblib.dump(scaler_y, config.SCALER_Y_PATH)
    print(f"\nTraining complete. Model and scalers saved to '{config.MODEL_DIR}'.")


def evaluate_model(characters_df, karts_df, mario_combos_df):
    """
    Evaluates the model using Leave-One-Out Cross-Validation (approximated with KFold).
    Returns the average Mean Absolute Error across all folds.
    """
    print("\n--- Evaluating model performance using Cross-Validation ---")
    
    common_karts = karts_df.index.intersection(mario_combos_df.index)
    kf = KFold(n_splits=len(common_karts)) # Leave-One-Out
    
    mario_stats = characters_df.loc['Mario'][['Speed', 'Acceleration', 'Weight', 'Handling']].values
    
    all_kart_names = np.array(common_karts)
    fold_errors = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(all_kart_names)):
        train_karts = all_kart_names[train_idx]
        val_kart = all_kart_names[val_idx][0] # The one kart we left out

        # --- Data Augmentation for this fold's training data ---
        X_train_list, y_train_list = [], []
        ref_characters = ['Mario', 'Bowser', 'Baby Peach']
        
        for kart_name in train_karts:
            kart_stats = karts_df.loc[kart_name].values
            for char_name in ref_characters:
                char_stats = characters_df.loc[char_name][['Speed', 'Acceleration', 'Weight', 'Handling']].values
                base_delta = char_stats - mario_stats
                X_train_list.append(list(kart_stats) + list(base_delta))
                y_train_list.append(base_delta * config.STAT_FACTOR)
        
        X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32)
        y_train = torch.tensor(np.array(y_train_list), dtype=torch.float32)

        scaler_X = StandardScaler().fit(X_train)
        X_train_scaled = torch.tensor(scaler_X.transform(X_train), dtype=torch.float32)
        scaler_y = StandardScaler().fit(y_train)
        y_train_scaled = torch.tensor(scaler_y.transform(y_train), dtype=torch.float32)

        # --- Train a temporary model for this fold ---
        model = StatPredictor()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
        
        for _ in range(config.EPOCHS):
            model.train()
            outputs = model(X_train_scaled)
            loss = criterion(outputs, y_train_scaled)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # --- Test on the left-out kart ---
        model.eval()
        
        # We test the prediction for Mario + the left-out kart
        val_kart_stats = karts_df.loc[val_kart].values
        # For Mario, the base_delta is zero
        val_input_features = torch.tensor([list(val_kart_stats) + [0,0,0,0]], dtype=torch.float32)
        val_input_scaled = torch.tensor(scaler_X.transform(val_input_features), dtype=torch.float32)

        with torch.no_grad():
            pred_scaled = model(val_input_scaled)
        
        # The model predicts the final delta. For Mario, this should be close to zero.
        pred_final_delta = scaler_y.inverse_transform(pred_scaled.numpy())
        
        # The true final delta for Mario is zero.
        true_final_delta = np.zeros((1, 4))
        
        # Calculate error
        mae = mean_absolute_error(true_final_delta, pred_final_delta)
        fold_errors.append(mae)
        print(f"Fold {fold+1}/{len(common_karts)}, Left-out Kart: {val_kart}, MAE: {mae:.4f}")

    avg_mae = np.mean(fold_errors)
    print(f"\n----------------------------------------------------")
    print(f"Cross-Validation Complete. Average MAE: {avg_mae:.4f}")
    print(f"----------------------------------------------------")
    return avg_mae


def optimize_hyperparameters(characters_df, karts_df, mario_combos_df):
    """
    Uses Optuna to find the best set of hyperparameters and STAT_FACTORS.
    """
    def objective(trial: optuna.Trial):
        # 1. Suggest hyperparameters for Optuna to test
        params = {
            'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
            'hidden_layer_1': trial.suggest_int('hidden_layer_1', 32, 128),
            'hidden_layer_2': trial.suggest_int('hidden_layer_2', 16, 64),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'epochs': trial.suggest_int('epochs', 500, 2000),
            'factor_speed': trial.suggest_float('factor_speed', 10.0, 25.0),
            'factor_accel': trial.suggest_float('factor_accel', -25.0, -10.0),
            'factor_weight': trial.suggest_float('factor_weight', 10.0, 20.0),
            'factor_handling': trial.suggest_float('factor_handling', 10.0, 20.0),
        }
        
        # 2. Run cross-validation with these specific parameters
        # (Ceci est une version simplifiée de la fonction evaluate_model)
        common_karts = karts_df.index.intersection(mario_combos_df.index)
        kf = KFold(n_splits=len(common_karts))
        mario_stats = characters_df.loc['Mario'][['Speed', 'Acceleration', 'Weight', 'Handling']].values
        all_kart_names = np.array(common_karts)
        fold_errors = []

        for train_idx, val_idx in kf.split(all_kart_names):
            # Setup train/val data for this fold
            train_karts = all_kart_names[train_idx]
            val_kart = all_kart_names[val_idx][0]
            
            # --- Data Augmentation for this fold's training data ---
            X_train_list, y_train_list = [], []
            ref_characters = ['Mario', 'Bowser', 'Toad']
            
            for kart_name in train_karts:
                kart_stats = karts_df.loc[kart_name].values
                for char_name in ref_characters:
                    char_stats = characters_df.loc[char_name][['Speed', 'Acceleration', 'Weight', 'Handling']].values
                    base_delta = char_stats - mario_stats
                    X_train_list.append(list(kart_stats) + list(base_delta))
                    
                    # Use factors suggested by Optuna for this trial
                    simulated_final_delta = np.array([
                        base_delta[0] * params['factor_speed'],
                        base_delta[1] * params['factor_accel'],
                        base_delta[2] * params['factor_weight'],
                        base_delta[3] * params['factor_handling']
                    ])
                    y_train_list.append(simulated_final_delta)
            
            X_train = torch.tensor(np.array(X_train_list), dtype=torch.float32)
            y_train = torch.tensor(np.array(y_train_list), dtype=torch.float32)

            scaler_X = StandardScaler().fit(X_train)
            X_train_scaled = torch.tensor(scaler_X.transform(X_train), dtype=torch.float32)
            scaler_y = StandardScaler().fit(y_train)
            y_train_scaled = torch.tensor(scaler_y.transform(y_train), dtype=torch.float32)
            
            # Temporary model with suggested architecture
            model = nn.Sequential(
                nn.Linear(config.INPUT_SIZE, params['hidden_layer_1']), nn.ReLU(), nn.Dropout(params['dropout']),
                nn.Linear(params['hidden_layer_1'], params['hidden_layer_2']), nn.ReLU(),
                nn.Linear(params['hidden_layer_2'], config.OUTPUT_SIZE)
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            criterion = nn.MSELoss()

            for _ in range(params['epochs']):
                model.train()
                outputs = model(X_train_scaled)
                loss = criterion(outputs, y_train_scaled)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            
            # Test on the left-out kart
            model.eval()
            val_kart_stats = karts_df.loc[val_kart].values
            val_input_features = torch.tensor([list(val_kart_stats) + [0,0,0,0]], dtype=torch.float32)
            val_input_scaled = torch.tensor(scaler_X.transform(val_input_features), dtype=torch.float32)
            with torch.no_grad():
                pred_scaled = model(val_input_scaled)
            pred_final_delta = scaler_y.inverse_transform(pred_scaled.numpy())
            mae = mean_absolute_error(np.zeros((1, 4)), pred_final_delta)
            fold_errors.append(mae)

        return np.mean(fold_errors)

    # 3. Create a study and run the optimization
    print("\n--- Starting Hyperparameter Optimization with Optuna ---")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100) # Run 100 different combinations

    print("\nOptimization finished!")
    print(f"Best trial MAE: {study.best_value}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    return study.best_params