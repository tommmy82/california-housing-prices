# 🏠 California Housing Prices – Projekt EDA

Projekt zaliczeniowy z eksploracyjnej analizy danych (EDA) w ramach przedmiotu SAD.

## 📁 Zawartość repozytorium

- `housing.csv` – zbiór danych (mediana cen domów w Kalifornii i cechy lokalizacji)
- `Zadanie zaliczeniowe SAD zwięźle.py` – skondensowany skrypt analizy danych
- `Zadanie zaliczeniowe końcowe SAD.py` – pełny skrypt z komentarzami i wizualizacjami

## 🎯 Cel projektu

Celem projektu było zbadanie, jakie czynniki wpływają na wartość domów w Kalifornii. Analiza objęła:

- czyszczenie danych (braki, outliery),
- imputację brakujących danych (porównanie średniej vs. mediany),
- test statystyczny ANOVA (ceny vs. `ocean_proximity`),
- inżynierię cech (`bedrooms_per_room`, `rooms_per_household`, itp.),
- wizualizacje (mapa cen, boxploty, heatmapa korelacji).

## 📊 Wnioski

- Dochód (`median_income`) jest silnie dodatnio skorelowany z ceną domu.
- Bliskość oceanu (`ocean_proximity`) istotnie wpływa na cenę – potwierdzone testem ANOVA.
- Mediana okazała się lepszą strategią imputacji niż średnia, z uwagi na obecność outlierów.
- Nowe cechy utworzone przez feature engineering dodały wartości predykcyjnej.

## 👨‍💻 Autor

Tomasz Kowalski, 2025