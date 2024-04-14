import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Sample size
n_samples = 1000

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

# Condition 1: Water level within 10 cm of the critical threshold and precipitation > 40 mm
df.loc[(df['water_level_distance'] <= 10) & (df['expected_precipitation'] > 10), 'flood'] = 1

# Condition 2: Very high turbidity (water opacity > 100 NTU) and high precipitation (> 60 mm)
df.loc[(df['water_opacity'] > 100) & (df['expected_precipitation'] > 60), 'flood'] = 1

# Condition 3: Moderate water level (within 30 cm of threshold) and very heavy rain (> 80 mm)
df.loc[(df['water_level_distance'] <= 30) & (df['expected_precipitation'] > 80), 'flood'] = 1


# Shuffle the DataFrame to mix up rows with flood and no flood scenarios
df = df.sample(frac=1).reset_index(drop=True)

# Save to CSV file
df.to_csv('flood_prediction_data.csv', index=False)

print(df.head())
print("Flood events in dataset:", df['flood'].sum())
