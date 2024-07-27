import streamlit as st
from surprise import Dataset, Reader, SVD
import pandas as pd
# import openai
import asyncio
from functools import lru_cache
# import os
import numpy as np

# Ensure the OpenAI API key is set in your environment variables
# api_key = os.getenv('OPENAI_API_KEY')
# openai.api_key = api_key

# Load the dataset
df = pd.read_csv('final_data.csv')

# Load collaborative filtering model
reader = Reader(rating_scale=(0, 100))
data = Dataset.load_from_df(df[['user_id', 'recipe_code', 'ratings']], reader)
trainset = data.build_full_trainset()
algo = SVD()
algo.fit(trainset)

# Define Streamlit app content
def streamlit_app():
    # Set page width and background color
    st.write(
        """
        <style>
        .stApp {
            background-image: url('https://images.pexels.com/photos/775032/pexels-photo-775032.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Create a Streamlit app
    st.title('Recipe Search App')

    # Create navigation sidebar with custom colors
    st.sidebar.title('Navigation')
    page = st.sidebar.radio('Go to', ['Home', 'About', 'Search'])

    # Display content based on selected page
    if page == 'Home':
        st.write('Welcome to Recipe Search App!')
        st.write('Use the navigation on the left to explore.')
    elif page == 'About':
        st.write('This app helps you search for recipes.')
        st.write('The app returns recipes based on ingredients. If a ingredints is not found, it  will return No recipes found with the given ingredients.')
        st.write('It uses a machine learning model to recommend recipes based on your input.')
    elif page == 'Search':
        st.title('Recipe Search')

        # Add ingredient search input
        ingredient_search = st.text_input('Search by Ingredients (comma separated):')
        ingredient_search_button = st.button('Search by Ingredients')

        # Initialize a session state to keep track of displayed recipe names
        if 'displayed_recipe_names' not in st.session_state:
            st.session_state.displayed_recipe_names = set()

        # Filter recipes based on ingredients and display the results
        if ingredient_search and ingredient_search_button:
            with st.spinner('Searching for recipes...'):
                ingredients = [ingredient.strip() for ingredient in ingredient_search.split(',')]
                filtered_recipes = df[df['ingredients'].str.contains('|'.join(ingredients), case=False, na=False)]

                if not filtered_recipes.empty:
                    recipe_codes = filtered_recipes['recipe_code'].tolist()
                    predictions = algo.test([(None, code, None) for code in recipe_codes])
                    predictions.sort(key=lambda x: x.est, reverse=True)

                    for prediction in predictions:
                        recipe_name = df.loc[df['recipe_code'] == prediction.iid]['recipe_name'].values[0]
                        if recipe_name not in st.session_state.displayed_recipe_names:
                            st.session_state.displayed_recipe_names.add(recipe_name)
                            st.markdown(f"**Recipe Name:** {recipe_name}")
                            st.write(f"**Predicted Rating:** {prediction.est}")
                            st.markdown(f"**Ingredients:** {df.loc[df['recipe_code'] == prediction.iid]['ingredients'].values[0]}")
                            st.markdown(f"**Cooking Instructions:** {df.loc[df['recipe_code'] == prediction.iid]['cooking_instructions'].values[0]}")
                            st.write('---')
                else:
                    st.write("No recipes found with the given ingredients.")

# Run the Streamlit app
if __name__ == "__main__":
    streamlit_app()
