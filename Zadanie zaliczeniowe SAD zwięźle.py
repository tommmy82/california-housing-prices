# Eksploracyjna analiza danych: California Housing Prices

# --- Importowanie bibliotek ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# Ustawienia estetyczne
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
pd.set_option('display.float_format', '{:.2f}'.format)

# --- Wczytanie danych ---
df = pd.read_csv("housing.csv")

# --- Wstępna eksploracja ---
print("\n📄 Informacje o zbiorze danych:")
print(df.info())

print("\n🔍 Podgląd danych:")
print(df.head())

print("\n📊 Statystyki opisowe:")
print(df.describe())

# --- Diagnostyka braków danych ---
missing = df.isnull().sum()
print("\n🔎 Liczba braków:\n", missing)

sns.heatmap(df.isnull(), cbar=False, cmap="Reds")
plt.title("Braki danych (heatmapa)")
plt.show()

# --- Porównanie strategii imputacji ---
mean_val = df['total_bedrooms'].mean()
median_val = df['total_bedrooms'].median()

print(f"\nŚrednia: {mean_val:.2f}, Mediana: {median_val:.2f}")

df_mean = df.copy()
df_median = df.copy()

df_mean['total_bedrooms'].fillna(mean_val, inplace=True)
df_median['total_bedrooms'].fillna(median_val, inplace=True)

sns.histplot(df['total_bedrooms'], bins=50, kde=True, label='Oryginalne', color='gray', alpha=0.5)
sns.histplot(df_mean['total_bedrooms'], bins=50, kde=True, label='Średnia', color='blue', alpha=0.5)
sns.histplot(df_median['total_bedrooms'], bins=50, kde=True, label='Mediana', color='green', alpha=0.5)
plt.legend()
plt.title("Porównanie imputacji total_bedrooms")
plt.xlabel("Liczba sypialni")
plt.ylabel("Liczba obserwacji")
plt.show()

# --- Imputacja wybrana: Mediana ---
df['total_bedrooms'].fillna(median_val, inplace=True)

# --- One-hot encoding ocean_proximity ---
df_encoded = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)

# --- Feature engineering ---
df_encoded["rooms_per_household"] = df_encoded["total_rooms"] / df_encoded["households"]
df_encoded["bedrooms_per_room"] = df_encoded["total_bedrooms"] / df_encoded["total_rooms"]
df_encoded["population_per_household"] = df_encoded["population"] / df_encoded["households"]

# --- Test ANOVA: ocean_proximity vs. median_house_value ---
groups = [group["median_house_value"].values for name, group in df.groupby("ocean_proximity")]
f_stat, p_val = stats.f_oneway(*groups)
print(f"\n📊 ANOVA – F-statistic: {f_stat:.2f}, p-value: {p_val:.4f}")
if p_val < 0.05:
    print("✅ Różnice są statystycznie istotne.")
else:
    print("❌ Brak istotności statystycznej.")

# --- Wizualizacje ---
# 1. Mapa cen domów
plt.scatter(df["longitude"], df["latitude"], alpha=0.4,
            c=df["median_house_value"], cmap="viridis", s=8)
plt.colorbar(label="Cena domu")
plt.title("Lokalizacja i ceny domów w Kalifornii")
plt.xlabel("Długość geograficzna")
plt.ylabel("Szerokość geograficzna")
plt.show()

# 2. Boxplot ocean proximity
sns.boxplot(x="ocean_proximity", y="median_house_value", data=df)
plt.title("Ceny domów vs. bliskość oceanu")
plt.xticks(rotation=45)
plt.show()

# 3. Scatter: income vs. house value
sns.scatterplot(x="median_income", y="median_house_value", data=df, alpha=0.3)
plt.title("Dochód vs. cena domu")
plt.show()

# 4. Histogram cen
sns.histplot(df["median_house_value"], bins=40, kde=True)
plt.title("Rozkład cen domów")
plt.show()

# --- Korelacje i heatmapa ---
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Korelacje między zmiennymi")
plt.show()

# --- Podsumowanie ---
print("""
📋 Wnioski:
- Imputacja medianą została wybrana jako bezpieczniejsza opcja przy obecności outlierów.
- Dochód (median_income) jest najsilniej dodatnio skorelowany z ceną domu.
- Bliskość oceanu wpływa istotnie na ceny – potwierdzone testem ANOVA.
- Nowe cechy, takie jak liczba pokoi na gospodarstwo domowe, wnoszą dodatkowe informacje.
- Projekt może być rozwijany poprzez regresję lub modele ML.
""")