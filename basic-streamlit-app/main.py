import streamlit as st
import pandas as pd

df = pd.read_csv("basic-streamlit-app/data/penguins.csv")  # Ensure the "data" folder exists with the CSV file

st.title("Penguins Dataset Exploration")

st.write("This app works to allow us to learn more about different penguins. Within this app, you can filter through the dataset of different penguins based on the island where they live, their species, as well as their body mass.")

st.write("Here's a look at our dataset:")
st.dataframe(df)

island = st.selectbox("Select an island:", df['island'].unique())
# Filtering the DataFrame based on user selection
filtered_df = df[df['island'] == island]
# Display the filtered results
st.dataframe(filtered_df)

species = st.selectbox("Select a species:", df['species'].unique())
# Filtering the DataFrame based on user selection
filtered_df2 = df[df['species'] == species]
# Display the filtered results
st.dataframe(filtered_df2)

mass = st.slider("Select a mass range:", 
                   min_value = df['body_mass_g'].min(), 
                   max_value = df['body_mass_g'].max())

# Filtering the DataFrame based on user selection


# Display the filtered results
st.write(f"Masses under {mass}:")
st.dataframe(df[df['body_mass_g'] <= mass])


#To run type streamlit run basic-streamlit-app/main.py in the terminal, make sure that the file is following the relative path