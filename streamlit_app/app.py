import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="🌍 Prédiction de la Consommation Énergétique", layout="wide")

# Chargement des données
@st.cache_data
def load_data():
    file_path = "../data/global_energy_consumption.csv"
    df = pd.read_csv(file_path)
    
    # Renommage des colonnes pour correspondre au PDF
    df.columns = ['Country', 'Year', 'Total_Consumption', 'Per_Capita_Use', 
                  'Renewable_Share', 'Fossil_Fuel_Dependency', 'Industrial_Use', 
                  'Household_Use', 'Carbon_Emissions', 'Energy_Price_Index']
    
    return df

df = load_data()

# Chargement du modèle et du scaler
rf_model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Nettoyage des valeurs aberrantes
cols_to_clean = ['Total_Consumption', 'Per_Capita_Use', 'Carbon_Emissions', 'Energy_Price_Index']

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    return df_filtered

for col in cols_to_clean:
    df = remove_outliers(df, col)

# Affichage des données après nettoyage
st.title("🌍 Prédiction de la Consommation Énergétique")
st.subheader("📊 Données après nettoyage")
st.write(df.head())

# Graphique de l'évolution de la consommation énergétique
st.subheader("📈 Évolution de la Consommation Énergétique")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=df["Year"], y=df["Total_Consumption"], ax=ax)
plt.xlabel("Année")
plt.ylabel("Consommation d'énergie (TWh)")
plt.title("Évolution de la Consommation Énergétique dans le Temps")
st.pyplot(fig)

# Sélection d'un pays
countries = df["Country"].unique()
selected_country = st.selectbox("🌎 Sélectionnez un pays :", countries)

# Filtrer les données pour le pays sélectionné
df_country = df[df["Country"] == selected_country]
st.subheader(f"📊 Données de {selected_country}")
st.write(df_country.head())

# Graphique de la consommation pour le pays sélectionné
st.subheader(f"📈 Évolution de la Consommation d'Énergie pour {selected_country}")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(x=df_country["Year"], y=df_country["Total_Consumption"], ax=ax)
plt.xlabel("Année")
plt.ylabel("Consommation d'énergie (TWh)")
plt.title(f"Consommation d'Énergie de {selected_country} dans le Temps")
st.pyplot(fig)

# Interface de Prédiction
st.subheader("🔮 Prédiction de la Consommation Énergétique")

# Sliders pour entrer les valeurs
per_capita_use = st.slider("Consommation par habitant (kWh)", min_value=0, max_value=50000, value=20000)
renewable_share = st.slider("Part des énergies renouvelables (%)", min_value=0, max_value=100, value=30)
fossil_dependency = st.slider("Dépendance aux énergies fossiles (%)", min_value=0, max_value=100, value=60)
energy_price = st.slider("Prix de l'énergie (USD/kWh)", min_value=0.01, max_value=1.00, value=0.10)

# Création de l'input sous forme de dataframe
input_data = pd.DataFrame([[per_capita_use, renewable_share, fossil_dependency, energy_price]], 
                          columns=['Per_Capita_Use', 'Renewable_Share', 
                                   'Fossil_Fuel_Dependency', 'Energy_Price_Index'])

# Normalisation des données
input_scaled = scaler.transform(input_data)

# Prédiction
prediction = rf_model.predict(input_scaled)

# Affichage du résultat
st.success(f"⚡ Consommation Énergétique Prédite : {prediction[0]:,.2f} TWh")