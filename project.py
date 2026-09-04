# Import required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
pd.set_option('display.max_columns', None)


# Load the dataset

df = pd.read_csv("seasonal_agriculture_performance_dataset.csv")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# Display first 5 rows

df.head()


# Dataset shape

print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())


# Examine dataset structure and data types

df.info()


# Display data types

df.dtypes


# Check missing values

missing_values = df.isnull().sum()

print("Missing values in each column:")
print(missing_values)

print("\nTotal missing values:", missing_values.sum())


# Separate numerical and categorical columns

numeric_columns = df.select_dtypes(include=np.number).columns
categorical_columns = df.select_dtypes(include='object').columns

# Fill numerical missing values using median
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical missing values using mode
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("Total missing values after handling:",
      df.isnull().sum().sum())


# Check duplicate records

print("Number of duplicate records:", df.duplicated().sum())


# Remove duplicate records

df = df.drop_duplicates()

print("Dataset shape after removing duplicates:", df.shape)
print("Duplicates remaining:", df.duplicated().sum())


# Statistical summary of numerical variables

df.describe()


# Summary of categorical variables

df.describe(include='object')


# Univariate: Crop distribution

plt.figure(figsize=(10, 6))

df['Crop'].value_counts().plot(kind='bar')

plt.title('Distribution of Crops')
plt.xlabel('Crop')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Univariate: Yield distribution

plt.figure(figsize=(9, 5))

sns.histplot(
    df['Yield_Tonnes_Ha'],
    bins=30,
    kde=True
)

plt.title('Distribution of Yield')
plt.xlabel('Yield (tonnes/ha)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Univariate: Profit distribution

plt.figure(figsize=(9, 5))

sns.histplot(
    df['Profit_INR'],
    bins=30,
    kde=True
)

plt.title('Distribution of Profit')
plt.xlabel('Profit (INR)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Univariate: Rainfall distribution

plt.figure(figsize=(9, 5))

sns.histplot(
    df['Rainfall_mm'],
    bins=30,
    kde=True
)

plt.title('Distribution of Rainfall')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()


# Univariate: Season distribution

plt.figure(figsize=(7, 7))

df['Season'].value_counts().plot(
    kind='pie',
    autopct='%1.1f%%',
    startangle=90,
    cmap='viridis'
)

plt.title('Distribution of Records by Season')
plt.ylabel('')

plt.tight_layout()
plt.show()


# Outlier investigation: Yield

plt.figure(figsize=(8, 5))

sns.boxplot(
    y=df['Yield_Tonnes_Ha']
)

plt.title('Outlier Investigation - Yield')
plt.ylabel('Yield (tonnes/ha)')

plt.tight_layout()
plt.show()


# Outlier investigation: Profit

plt.figure(figsize=(8, 5))

sns.boxplot(
    y=df['Profit_INR']
)

plt.title('Outlier Investigation - Profit')
plt.ylabel('Profit (INR)')

plt.tight_layout()
plt.show()


# Outlier investigation: Water usage

plt.figure(figsize=(8, 5))

sns.boxplot(
    y=df['Water_Used_m3']
)

plt.title('Outlier Investigation - Water Usage')
plt.ylabel('Water Used (m³)')

plt.tight_layout()
plt.show()


# Bivariate: Rainfall and Yield

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='Rainfall_mm',
    y='Yield_Tonnes_Ha',
    hue='Season'
)

plt.title('Rainfall vs Yield')
plt.xlabel('Rainfall (mm)')
plt.ylabel('Yield (tonnes/ha)')

plt.tight_layout()
plt.show()


# Bivariate: Temperature and Yield

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='Avg_Temperature_C',
    y='Yield_Tonnes_Ha',
    hue='Season'
)

plt.title('Temperature vs Yield')
plt.xlabel('Average Temperature (°C)')
plt.ylabel('Yield (tonnes/ha)')

plt.tight_layout()
plt.show()


# Bivariate: Water Usage and Yield

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='Water_Used_m3',
    y='Yield_Tonnes_Ha',
    hue='Season'
)

plt.title('Water Usage vs Yield')
plt.xlabel('Water Used (m³)')
plt.ylabel('Yield (tonnes/ha)')

plt.tight_layout()
plt.show()


# Bivariate: Fertilizer and Yield

plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='Fertilizer_kg_ha',
    y='Yield_Tonnes_Ha',
    hue='Season'
)

plt.title('Fertilizer Usage vs Yield')
plt.xlabel('Fertilizer (kg/ha)')
plt.ylabel('Yield (tonnes/ha)')

plt.tight_layout()
plt.show()


# Multivariate: Pair plot

selected_columns = [
    'Avg_Temperature_C',
    'Rainfall_mm',
    'Soil_Moisture_pct',
    'Fertilizer_kg_ha',
    'Pesticide_Litre_ha',
    'Yield_Tonnes_Ha'
]

sns.pairplot(
    df[selected_columns + ['Season']],
    hue='Season',
    diag_kind='kde'
)

plt.suptitle(
    'Pair Plot of Environmental and Performance Metrics by Season',
    y=1.02
)

plt.show()


# Correlation matrix

correlation = df.select_dtypes(
    include=np.number
).corr()

correlation


# Correlation heatmap

plt.figure(figsize=(15, 11))

sns.heatmap(
    correlation,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=0.5
)

plt.title('Correlation Heatmap of Agricultural Variables')

plt.tight_layout()
plt.show()


# Correlation with Yield

yield_correlation = (
    correlation['Yield_Tonnes_Ha']
    .sort_values(ascending=False)
)

print("Correlation with Yield:")
print(yield_correlation)


# Create seasonal summary

season_summary = df.groupby('Season').agg({

    'Yield_Tonnes_Ha': 'mean',
    'Production_Tonnes': 'mean',
    'Revenue_INR': 'mean',
    'Profit_INR': 'mean',
    'Water_Used_m3': 'mean',
    'Water_Efficiency_t_per_1000m3': 'mean',
    'Rainfall_mm': 'mean',
    'Avg_Temperature_C': 'mean',
    'Soil_Moisture_pct': 'mean'

})

season_summary


# Seasonal comparison: Yield

plt.figure(figsize=(8, 5))

df.groupby('Season')['Yield_Tonnes_Ha'].mean().plot(
    kind='bar'
)

plt.title('Average Yield by Season')
plt.xlabel('Season')
plt.ylabel('Average Yield (tonnes/ha)')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Seasonal comparison: Production

plt.figure(figsize=(8, 5))

df.groupby('Season')['Production_Tonnes'].mean().plot(
    kind='bar'
)

plt.title('Average Production by Season')
plt.xlabel('Season')
plt.ylabel('Average Production (tonnes)')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Seasonal comparison: Profit

plt.figure(figsize=(8, 5))

df.groupby('Season')['Profit_INR'].mean().plot(
    kind='bar'
)

plt.title('Average Profit by Season')
plt.xlabel('Season')
plt.ylabel('Average Profit (INR)')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Seasonal comparison: Revenue

plt.figure(figsize=(8, 5))

df.groupby('Season')['Revenue_INR'].mean().plot(
    kind='bar'
)

plt.title('Average Revenue by Season')
plt.xlabel('Season')
plt.ylabel('Average Revenue (INR)')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Seasonal comparison: Water usage

plt.figure(figsize=(8, 5))

df.groupby('Season')['Water_Used_m3'].mean().plot(
    kind='bar'
)

plt.title('Average Water Usage by Season')
plt.xlabel('Season')
plt.ylabel('Water Used (m³)')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Seasonal comparison: Water efficiency

plt.figure(figsize=(8, 5))

df.groupby('Season')[
    'Water_Efficiency_t_per_1000m3'
].mean().plot(kind='bar')

plt.title('Average Water Efficiency by Season')
plt.xlabel('Season')
plt.ylabel('Tonnes per 1000 m³')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Crop-wise average yield

crop_yield = (
    df.groupby('Crop')['Yield_Tonnes_Ha']
    .mean()
    .sort_values(ascending=False)
)

print(crop_yield)


# Crop-wise yield visualization

plt.figure(figsize=(10, 6))

crop_yield.plot(kind='bar')

plt.title('Average Yield by Crop')
plt.xlabel('Crop')
plt.ylabel('Average Yield (tonnes/ha)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Top 10 crops by average profit

top_crops_profit = (
    df.groupby('Crop')['Profit_INR']
    .mean()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))

top_crops_profit.plot(kind='bar')

plt.title('Top 10 Crops by Average Profit')
plt.xlabel('Crop')
plt.ylabel('Average Profit (INR)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# State-wise average yield

state_yield = (
    df.groupby('State')['Yield_Tonnes_Ha']
    .mean()
    .sort_values(ascending=False)
)

print(state_yield)


# State-wise yield visualization

plt.figure(figsize=(12, 6))

state_yield.plot(kind='bar')

plt.title('Average Yield by State')
plt.xlabel('State')
plt.ylabel('Average Yield (tonnes/ha)')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Student-designed analysis 1:
# Most profitable season

profit_by_season = (
    df.groupby('Season')['Profit_INR']
    .mean()
    .sort_values(ascending=False)
)

print("Average Profit by Season:")
print(profit_by_season)

print(
    "\nMost profitable season:",
    profit_by_season.idxmax()
)

print(
    "Average profit:",
    round(profit_by_season.max(), 2)
)


# Student-designed analysis 2:
# Highest yielding crop

crop_yield = (
    df.groupby('Crop')['Yield_Tonnes_Ha']
    .mean()
    .sort_values(ascending=False)
)

print("Average Yield by Crop:")
print(crop_yield)

print(
    "\nHighest yielding crop:",
    crop_yield.idxmax()
)

print(
    "Average yield:",
    round(crop_yield.max(), 2)
)


# Student-designed analysis 3:
# Best water-efficient season

water_efficiency = (
    df.groupby('Season')
    ['Water_Efficiency_t_per_1000m3']
    .mean()
    .sort_values(ascending=False)
)

print("Water Efficiency by Season:")
print(water_efficiency)

print(
    "\nMost water-efficient season:",
    water_efficiency.idxmax()
)

print(
    "Average water efficiency:",
    round(water_efficiency.max(), 2)
)


# Disease and pest risk by season

risk_by_season = (
    df.groupby('Season')
    ['Disease_Pest_Risk_pct']
    .mean()
    .sort_values(ascending=False)
)

print(risk_by_season)


plt.figure(figsize=(8, 5))

risk_by_season.plot(kind='bar')

plt.title('Average Disease and Pest Risk by Season')
plt.xlabel('Season')
plt.ylabel('Disease/Pest Risk (%)')
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# Important results for findings

best_yield_season = (
    df.groupby('Season')['Yield_Tonnes_Ha']
    .mean()
    .idxmax()
)

best_profit_season = (
    df.groupby('Season')['Profit_INR']
    .mean()
    .idxmax()
)

best_water_season = (
    df.groupby('Season')
    ['Water_Efficiency_t_per_1000m3']
    .mean()
    .idxmax()
)

best_crop = (
    df.groupby('Crop')['Yield_Tonnes_Ha']
    .mean()
    .idxmax()
)

best_state = (
    df.groupby('State')['Yield_Tonnes_Ha']
    .mean()
    .idxmax()
)

print("Best season by yield:", best_yield_season)
print("Most profitable season:", best_profit_season)
print("Most water-efficient season:", best_water_season)
print("Highest yielding crop:", best_crop)
print("Highest yielding state:", best_state)