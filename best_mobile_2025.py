"""
Best Mobile 2025 - Mobile Phone Ranking Analysis
This script analyzes mobile phone specifications and creates a comprehensive ranking
based on performance, camera quality, display, battery life, and price.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the mobile phone dataset
df = pd.read_csv('mobiles.csv')

# Display basic information about the dataset
print("Dataset Info:")
df.info()

# Shows count of missing values per column (NaNs). Use this to check for gaps before computing scores.
print("\nMissing Values:")
print(df.isna().sum())

# Compute a weighted performance score combining CPU speed, CPU cores, and a general spec score.
# Weights chosen to reflect relative importance:
# - cpu_speed: 40% (higher clock speed improves single-threaded performance)
# - cpu_cores: 20% (more cores help multi-threaded tasks)
# - spec_score: 40% (an aggregated spec metric included by dataset)
df['performance_score'] = (
    df['cpu_speed']*0.4 +
    df['cpu_cores']*0.2 +
    df['spec_score']*0.4
)

print("\nPerformance Score:")
print(df['performance_score'])

# Compute camera score by weighting rear and front camera primary sensor values.
# Rear camera usually contributes more to overall photo quality, so it gets higher weight (70%).
df['camera_score'] = (
    df['rear_primary']*0.7 +
    df['front_primary']*0.3
)

print("\nCamera Score:")
print(df['camera_score'])

# Combine refresh rate and pixel density (PPI) into a display score.
# PPI (sharpness) is weighted 60% and refresh rate 40%; adjust if you prefer smoother motion to be more important.
df['display_score'] = (
    df['refresh_rate']*0.4 +
    df['ppi']*0.6
)

print("\nDisplay Score:")
print(df['display_score'])

# Normalize selected features to [0,1] so different units don't dominate the weighted final score.
# We scale: performance_score, camera_score, display_score, battery, price.
# Note: price will be inverted later (1 - price_n) because lower price is better.
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[['performance_score','camera_score','display_score','battery','price']])
scaled_df = pd.DataFrame(scaled, columns=['perf_n','cam_n','disp_n','battery_n','price_n'])

# Attach normalized columns back to the original dataframe.
df = pd.concat([df, scaled_df], axis=1)

print("\nNormalized Data:")
print(df)

# Aggregate normalized scores into a single final score using specified weights:
# - perf_n: 35% (performance)
# - cam_n: 25% (camera)
# - disp_n: 20% (display)
# - battery_n: 10% (battery life)
# - price_n: 10% but inverted as (1 - price_n) so that a lower price increases the final score
df['final_score'] = (
    df['perf_n']*0.35 +
    df['cam_n']*0.25 +
    df['disp_n']*0.20 +
    df['battery_n']*0.10 +
    (1 - df['price_n'])*0.10   # lower price = better
)

print("\nFinal Score:")
print(df['final_score'])

# Get the top 10 phones sorted by final score
best = df.sort_values('final_score', ascending=False)
print("\nTop 10 Best Phones:")
print(best.head(10))

# Get details of the best phone
print("\nBest Phone Overall:")
print(best.iloc[0])

# Create a visualization of top 10 phones
plt.figure(figsize=(10, 5))
plt.bar(best['name'][:10], best['final_score'][:10])
plt.xticks(rotation=75)
plt.title("Top Phones by Final Score")
plt.ylabel("Score")
plt.tight_layout()
plt.show()

print("\nAnalysis complete!")
