import pandas as pd

# Wczytaj dane z lokalnego pliku lub podaj ścieżkę do pliku CSV
df = pd.read_csv('housing.csv')

import seaborn as sns
import matplotlib.pyplot as plt

# Liczba braków
missing_values = df.isnull().sum()
print("🔎 Liczba braków:\n", missing_values)

# Wizualizacja braków (heatmapa)
plt.figure(figsize=(8, 4))
sns.heatmap(df.isnull(), cbar=False, cmap="Reds")
plt.title("Braki danych (heatmapa)")
plt.show()

mean_bedrooms = df['total_bedrooms'].mean()
median_bedrooms = df['total_bedrooms'].median()

print(f"Średnia liczba sypialni: {mean_bedrooms:.2f}")
print(f"Mediana liczby sypialni: {median_bedrooms:.2f}")

# Kopie do porównania
df_mean_imputed = df.copy()
df_median_imputed = df.copy()

df_mean_imputed['total_bedrooms'].fillna(mean_bedrooms, inplace=True)
df_median_imputed['total_bedrooms'].fillna(median_bedrooms, inplace=True)

# Histogram porównawczy
plt.figure(figsize=(10, 6))
sns.histplot(df['total_bedrooms'], bins=50, kde=True, label='Oryginalne (z brakami)', color='gray', alpha=0.5)
sns.histplot(df_mean_imputed['total_bedrooms'], bins=50, kde=True, label='Imputacja średnią', color='blue', alpha=0.5)
sns.histplot(df_median_imputed['total_bedrooms'], bins=50, kde=True, label='Imputacja medianą', color='green', alpha=0.5)

plt.title("Porównanie metod imputacji dla total_bedrooms")
plt.xlabel("Liczba sypialni")
plt.ylabel("Liczba obserwacji")
plt.legend()
plt.show()

median_value = df["total_bedrooms"].median()
df["total_bedrooms"].fillna(median_value, inplace=True)

from scipy.stats import zscore

# Wybierz kolumny liczbowe
numerics = df.select_dtypes(include='number')
z_scores = zscore(numerics)

# Maski dla outlierów (z-score > 3)
outliers = (abs(z_scores) > 3)
outlier_counts = outliers.sum()
print("🚨 Liczba outlierów (> 3 std):\n", outlier_counts)

sns.boxplot(x=df['median_house_value'])
plt.title("Boxplot wartości domów")
plt.show()

import scipy.stats as stats

# Grupy cen według kategorii ocean_proximity
groups = [group["median_house_value"].values for name, group in df.groupby("ocean_proximity")]

# Test ANOVA
f_stat, p_val = stats.f_oneway(*groups)

print(f"\n📊 ANOVA – F-statistic: {f_stat:.2f}, p-value: {p_val:.4f}")

# Interpretacja automatyczna
if p_val < 0.05:
    print("✅ Różnice średnich cen domów pomiędzy kategoriami 'ocean_proximity' są statystycznie istotne.")
else:
    print("❌ Nie ma statystycznie istotnych różnic średnich cen między kategoriami.")

# 🔷 Pomysł A: One-hot encoding dla zmiennej jakościowej 'ocean_proximity'
df_encoded = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# 🔷 Pomysł B: Feature engineering – tworzenie nowych zmiennych
df_encoded["rooms_per_household"] = df_encoded["total_rooms"] / df_encoded["households"]
df_encoded["bedrooms_per_room"] = df_encoded["total_bedrooms"] / df_encoded["total_rooms"]
df_encoded["population_per_household"] = df_encoded["population"] / df_encoded["households"]

# Podgląd nowych kolumn
print("\n🆕 Nowe kolumny (feature engineering):")
print(df_encoded[["rooms_per_household", "bedrooms_per_room", "population_per_household"]].describe())

print("\n📊 Korelacja nowych zmiennych z ceną domu:")
corrs = df_encoded[[
    "median_house_value", "rooms_per_household", "bedrooms_per_room", "population_per_household"
]].corr()

print(corrs["median_house_value"].sort_values(ascending=False))

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(df_encoded.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("📌 Korelacja między zmiennymi liczbowymi (po feature engineering)")
plt.show()

# Wizualizacja rozmieszczenia domów względem ich wartości (median_house_value) na mapie Kalifornii:


plt.figure(figsize=(10, 6))
plt.scatter(df["longitude"], df["latitude"], alpha=0.4,
            c=df["median_house_value"], cmap="viridis", s=8)
plt.colorbar(label="Mediana ceny domu ($)")
plt.xlabel("Długość geograficzna")
plt.ylabel("Szerokość geograficzna")
plt.title("📍 Lokalizacja i ceny domów w Kalifornii")
plt.show()

# Wykresy kategorii ocean_proximity a ceny

plt.figure(figsize=(8, 5))
sns.boxplot(x="ocean_proximity", y="median_house_value", data=df)
plt.title("Ceny domów a odległość od oceanu")
plt.xticks(rotation=45)
plt.ylabel("Mediana wartości domu")
plt.show()

# Związek dochodu i wartości domu

plt.figure(figsize=(8, 5))
sns.scatterplot(x="median_income", y="median_house_value", data=df, alpha=0.3)
plt.title("Związek dochodu z wartością domu")
plt.xlabel("Mediana dochodu (w dziesiątkach tys. $)")
plt.ylabel("Mediana wartości domu ($)")
plt.show()

# Histogram mediany cen

sns.histplot(df["median_house_value"], bins=40, kde=True)
plt.title("Rozkład cen domów w Kalifornii")
plt.xlabel("Mediana wartości domu")
plt.ylabel("Liczba obserwacji")
plt.show()

# Analiza korelacji z heatmapą

## Korelacja między wszystkimi zmiennymi liczbowymi:

corr_matrix = df_encoded.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("🔗 Korelacja między zmiennymi liczbowymi")
plt.show()

### Top 5 zmiennych najbardziej skorelowanych z ceną domu

top_corrs = corr_matrix["median_house_value"].drop("median_house_value").sort_values(ascending=False)
print("📊 Top 5 korelacji z ceną domu:\n", top_corrs.head(5))



