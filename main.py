# main.py
import argparse, os
import pandas as pd

import config
from model import train_model, evaluate_model, optimize_hyperparameters
from optimizer import MKWorldOptimizer

def setup_directories():
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

def load_data_from_csv() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading data from CSV files...")
    try:
        characters_df = pd.read_csv(config.CHARACTERS_PATH, index_col='Character')
        karts_df = pd.read_csv(config.KARTS_PATH, index_col='Name')
        mario_combos_df = pd.read_csv(config.MARIO_COMBOS_PATH, index_col='Name')
    except FileNotFoundError as e:
        print(f"Error: {e}. Make sure CSV files are in the '{config.DATA_DIR}' directory.")
        exit()
    return characters_df, karts_df, mario_combos_df

def main():
    setup_directories()
    
    parser = argparse.ArgumentParser(
        description="Mario Kart World Build Optimizer - A tool for finding the best builds.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # --- Actions ---
    actions = parser.add_argument_group('Actions')
    actions.add_argument('--train', action='store_true', help="Train the model.")
    actions.add_argument('--force-retrain', action='store_true', help="Force the model to retrain.")
    actions.add_argument('--predict', nargs=2, metavar=('CHARACTER', 'KART'), help="Predict stats for a specific combo.")
    actions.add_argument('--search', action='store_true', help="Search for the best builds using weights or a profile.")
    actions.add_argument('--heatmap', action='store_true', help="Generate a heatmap of combination scores.")
    actions.add_argument('--evaluate', action='store_true', help="Evaluate model performance using cross-validation.")
    actions.add_argument('--optimize', action='store_true', help="Automatically find the best hyperparameters using Optuna.")

    # --- Search Parameters ---
    params = parser.add_argument_group('Search Parameters')
    params.add_argument('--profile', choices=config.SEARCH_PROFILES.keys(), help="Use a pre-defined search profile.")
    params.add_argument('--speed', type=float, help="Custom weight for Speed.")
    params.add_argument('--accel', type=float, help="Custom weight for Acceleration.")
    params.add_argument('--weight', type=float, help="Custom weight for Weight.")
    params.add_argument('--handling', type=float, help="Custom weight for Handling.")
    params.add_argument('--top_n', type=int, default=10, help="Number of results to show for --search.")

    args = parser.parse_args()
    
    characters_df, karts_df, mario_combos_df = load_data_from_csv()

    if args.train or args.force_retrain:
        train_model(characters_df, karts_df, mario_combos_df, force_retrain=args.force_retrain)

    if args.evaluate:
        evaluate_model(characters_df, karts_df, mario_combos_df)

    if args.predict or args.search or args.heatmap:
        optimizer = MKWorldOptimizer(characters_df, karts_df, mario_combos_df)

        if args.predict:
            char, kart = args.predict
            if char not in optimizer.characters_df.index:
                print(f"Error: Character '{char}' not found. Check spelling and capitalization.")
                return
            if kart not in optimizer.karts_df.index:
                print(f"Error: Kart '{kart}' not found. Use lowercase, no spaces.")
                return
            prediction = optimizer.predict_single_combo(char, kart)
            print(f"\n--- Predicted Stats for {char} + {kart} ---")
            print(pd.DataFrame([prediction]))
        
        weights = {}
        if args.profile:
            weights = config.SEARCH_PROFILES[args.profile]
        else: # Build weights from individual arguments
            custom_weights = {
                'Speed': args.speed, 'Acceleration': args.accel,
                'Weight': args.weight, 'Handling': args.handling,
            }
            # Only include weights that were actually provided
            weights = {k: v for k, v in custom_weights.items() if v is not None}

        if not weights and (args.search or args.heatmap):
             print("\nNo weights or profile provided. Using default 'balanced' profile.")
             weights = config.SEARCH_PROFILES['balanced']

        if args.search:
            optimizer.find_best_combos(weights, args.top_n)

        if args.heatmap:
            optimizer.generate_heatmap(weights)
    
    if args.optimize:
        best_params = optimize_hyperparameters(characters_df, karts_df, mario_combos_df)
        print("\nTo use these new parameters, update your config.py file and retrain the model with --force-retrain.")

    if not any(vars(args).values()):
        parser.print_help()

if __name__ == "__main__":
    main()