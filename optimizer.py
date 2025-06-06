# optimizer.py
import os
import pandas as pd
import torch
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Import from our custom modules
import config
from model import StatPredictor

class MKWorldOptimizer:
    def __init__(self, characters_df, karts_df, mario_combos_df):
        self.characters_df = characters_df
        self.karts_df = karts_df
        self.mario_combos_df = mario_combos_df
        self.stat_columns = ['Speed', 'Acceleration', 'Weight', 'Handling']
        
        try:
            self.model = StatPredictor()
            self.model.load_state_dict(torch.load(config.MODEL_PATH))
            self.model.eval()
            self.scaler_X = joblib.load(config.SCALER_X_PATH)
            self.scaler_y = joblib.load(config.SCALER_Y_PATH)
        except FileNotFoundError:
            print("Error: Model or scalers not found. Please train the model first.")
            exit()
        
        self.mario_base_stats = self.characters_df.loc['Mario'][self.stat_columns].values
        self.all_combos_df = self._generate_all_combos()

    def predict_single_combo(self, character_name: str, kart_name: str) -> dict:
        char_stats = self.characters_df.loc[character_name][self.stat_columns].values
        kart_stats = self.karts_df.loc[kart_name][self.stat_columns].values
        
        base_delta = char_stats - self.mario_base_stats
        input_features = torch.tensor([list(kart_stats) + list(base_delta)], dtype=torch.float32)
        input_scaled = torch.tensor(self.scaler_X.transform(input_features), dtype=torch.float32)
        
        with torch.no_grad():
            prediction_scaled = self.model(input_scaled)
        
        predicted_final_delta = self.scaler_y.inverse_transform(prediction_scaled.numpy())
        
        if kart_name not in self.mario_combos_df.index:
            return {stat: 0 for stat in self.stat_columns}
            
        mario_kart_final_stats = self.mario_combos_df.loc[kart_name].values
        final_prediction = mario_kart_final_stats + predicted_final_delta[0]
        
        return {stat: round(val) for stat, val in zip(self.stat_columns, final_prediction)}
    
    def _generate_all_combos(self) -> pd.DataFrame:
        print("\nPre-calculating all possible combinations...")
        all_combos_data = []
        for char_name in self.characters_df.index:
            for kart_name in self.karts_df.index:
                if kart_name not in self.mario_combos_df.index:
                    continue
                
                predicted_stats = self.predict_single_combo(char_name, kart_name)
                display_kart_name = kart_name.replace('-', ' ').title()
                combo_info = {'Character': char_name, 'Kart': display_kart_name, **predicted_stats}
                all_combos_data.append(combo_info)
        df = pd.DataFrame(all_combos_data)
        df.to_csv(config.ALL_COMBOS_PATH, index=False)
        print(f"Saved all {len(df)} valid combinations to '{config.ALL_COMBOS_PATH}'.")
        return df

    def find_best_combos(self, weights: dict, top_n: int = 10):
        print(f"\n--- Finding Top {top_n} Builds with weights: {weights} ---")
        temp_df = self.all_combos_df.copy()
        
        temp_df['Score'] = 0
        for stat, weight in weights.items():
            if stat in temp_df.columns and weight != 0:
                temp_df['Score'] += temp_df[stat] * weight
        
        result_df = temp_df.sort_values(by='Score', ascending=False).head(top_n)
        print(result_df.to_string())
        
        filename = f"top_{top_n}_builds_{'_'.join(k for k,v in weights.items() if v != 0)}.csv"
        path = os.path.join(config.OUTPUT_DIR, filename)
        result_df.to_csv(path, index=False)
        print(f"\nResults saved to '{path}'")

    def generate_heatmap(self, weights: dict):
        print(f"\n--- Generating Heatmap with weights: {weights} ---")
        temp_df = self.all_combos_df.copy()
        
        temp_df['Score'] = 0
        for stat, weight in weights.items():
            if stat in temp_df.columns:
                temp_df['Score'] += temp_df[stat] * weight
        
        heatmap_data = temp_df.pivot(index='Character', columns='Kart', values='Score')
        
        sorted_characters = self.characters_df.sort_values('Class').index
        heatmap_data = heatmap_data.reindex(index=sorted_characters)
        
        plt.figure(figsize=(20, 15))
        sns.heatmap(heatmap_data, cmap="viridis", annot=False)
        plt.title(f'Combination Scores (Weights: {weights})', fontsize=20)
        plt.xlabel('Kart', fontsize=15)
        plt.ylabel('Character', fontsize=15)
        plt.tight_layout()
        plt.savefig(config.HEATMAP_PATH)
        print(f"Heatmap saved to '{config.HEATMAP_PATH}'.")