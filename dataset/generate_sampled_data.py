import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

original_path = Path("dataset","earthquake_data.csv")
output_path = Path("dataset","sample.csv")

df = pd.read_csv(original_path)

class_ratios = df['alert'].value_counts(normalize=True)
print(f"Original class distribution:\n{class_ratios}\n")

sample_fraction = 0.05

sampled_dfs = []
for alert_class in df['alert'].unique():
    class_df = df[df['alert'] == alert_class]
    n_samples = max(1, int(len(class_df) * sample_fraction))
    sampled = class_df.sample(n=n_samples, random_state=42)
    sampled_dfs.append(sampled)

sampled_df = pd.concat(sampled_dfs, ignore_index=True)

numeric_cols = ['magnitude', 'depth', 'cdi', 'mmi', 'sig']
perturbation_factors = {
    'magnitude': 0.05,
    'depth': 0.1,
    'cdi': 0.15,
    'mmi': 0.15,
    'sig': 0.2
}

for col in numeric_cols:
    noise = sampled_df[col] * np.random.uniform(-perturbation_factors[col], perturbation_factors[col], size=len(sampled_df))
    sampled_df[col] = sampled_df[col] + noise
    if col in ['magnitude', 'depth', 'cdi', 'mmi']:
        sampled_df[col] = sampled_df[col].round(1)
    else:
        sampled_df[col] = sampled_df[col].round(0)

sampled_df_out = sampled_df.drop(columns=["alert"])
sampled_df_out.to_csv(output_path, index=False)

new_class_ratios = sampled_df['alert'].value_counts(normalize=True)
print(f"New dataset class distribution:\n{new_class_ratios}\n")
print(f"Original samples: {len(df)}, New samples: {len(sampled_df)}")
print(f"Saved to: {output_path}")