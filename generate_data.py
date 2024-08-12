import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Sample size
n_samples = 1000
sequence_length = 5  # Number of timesteps in each sequence

# Generate random data
data = {
    'water_level_distance': np.random.normal(loc=50, scale=20, size=n_samples),  # cm from the critical level
    'water_opacity': np.random.uniform(0, 150, size=n_samples),  # NTU
    'expected_precipitation': np.random.uniform(0, 100, size=n_samples),  # mm of rain expected
    'flood': np.zeros(n_samples)  # No flood by default
}

df = pd.DataFrame(data)

# Conditions for flooding
df.loc[(df['water_level_distance'] >= 1) & (df['expected_precipitation'] == 0), 'flood'] = 0

# Condition 1: Water level within 10 cm of the critical threshold and precipitation > 10 mm
df.loc[(df['water_level_distance'] <= 10) & (df['expected_precipitation'] > 10), 'flood'] = 1

# Condition 2: Very high turbidity (water opacity > 100 NTU) and high precipitation (> 60 mm)
df.loc[(df['water_opacity'] > 100) & (df['expected_precipitation'] > 60), 'flood'] = 1

# Condition 3: Moderate water level (within 30 cm of threshold) and very heavy rain (> 80 mm)
df.loc[(df['water_level_distance'] <= 30) & (df['expected_precipitation'] > 80), 'flood'] = 1

# Ensure sequential data by creating sequences
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:i + seq_length][['water_level_distance', 'water_opacity', 'expected_precipitation']]
        y = data.iloc[i + seq_length - 1]['flood']
        xs.append(x.values)
        ys.append(y)
    return np.array(xs), np.array(ys)

X, y = create_sequences(df, sequence_length)

# Save the sequences to a single file
np.savez('flood_prediction_sequences_5.npz', X=X, y=y)

print(df.head())
print("Flood events in dataset:", df['flood'].sum())
print(f"Total sequences: {len(X)}")
