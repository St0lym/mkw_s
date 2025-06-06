# config.py
import os

# --- DIRECTORIES ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')

# --- FILE PATHS ---
CHARACTERS_PATH = os.path.join(DATA_DIR, 'characters.csv')
KARTS_PATH = os.path.join(DATA_DIR, 'karts.csv')
MARIO_COMBOS_PATH = os.path.join(DATA_DIR, 'mario_kart_combos_100.csv')
ALL_COMBOS_PATH = os.path.join(OUTPUT_DIR, 'all_combinations_stats.csv')
HEATMAP_PATH = os.path.join(OUTPUT_DIR, 'combinations_heatmap.png')

MODEL_PATH = os.path.join(MODEL_DIR, 'stat_predictor.pth')
SCALER_X_PATH = os.path.join(MODEL_DIR, 'scaler_X.pkl')
SCALER_Y_PATH = os.path.join(MODEL_DIR, 'scaler_Y.pkl')

# --- MODEL HYPERPARAMETERS ---
INPUT_SIZE = 8
OUTPUT_SIZE = 4
HIDDEN_LAYER_1 = 128
HIDDEN_LAYER_2 = 64
DROPOUT_RATE = 0.2

# --- TRAINING HYPERPARAMETERS ---
LEARNING_RATE = 0.005
EPOCHS = 1000
# Facteur de conversion hypothétique pour l'augmentation de données.
# C'est l'hypothèse clé : une différence de 1.0 en stat de base
# équivaut à une différence de ~15.0 sur l'échelle de 100.
STAT_FACTOR = 25.0

# --- PRE-DEFINED SEARCH PROFILES ---
SEARCH_PROFILES = {
    "balanced": {'Speed': 1.0, 'Acceleration': 1.0, 'Handling': 0.5, 'Weight': 0.2},
    "max_speed": {'Speed': 1.0, 'Acceleration': -0.2}, # Pénalise un peu la faible accélération
    "max_accel": {'Acceleration': 1.0, 'Handling': 0.5},
    "max_handling": {'Handling': 1.0, 'Acceleration': 0.5},
}