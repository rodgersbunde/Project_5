    import streamlit as st
    from surprise import Dataset, Reader, SVD
    import pandas as pd
    import openai
    from functools import lru_cache
    import numpy as np
    from tenacity import retry, stop_after_attempt, wait_exponential
    from ratelimit import limits, sleep_and_retry
    import logging
    import time
    import os
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Configure OpenAI
    try:
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key:
            st.error("Please set your OpenAI API key in the .env file")
            logger.error("API key not found in environment variables")
            st.stop()
    except Exception as e:
        st.error("Error configuring OpenAI API key")
        logger.error(f"API key configuration error: {str(e)}")
        st.stop()

    # Rate limiting decorators
    CALLS_PER_MINUTE = 3  # Reduced from 50 to 3 to be more conservative
    PERIOD = 60

    @sleep_and_retry
    @limits(calls=CALLS_PER_MINUTE, period=PERIOD)
    @retry(wait=wait_exponential(multiplier=5, min=10, max=30),  # Increased wait times
        stop=stop_after_attempt(2))  # Reduced attempts to 2
    def rate_limited_api_call(recipe_name):
        """
        Make rate-limited API calls to OpenAI with exponential backoff
        """
        try:
            logger.info(f"Making API call for recipe: {recipe_name}")
            response = openai.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful cooking assistant. When given a recipe name, provide the ingredients and cooking instructions in a structured format. Always format your response with 'Ingredients:' and 'Instructions:' sections."
                    },
                    {
                        "role": "user",
                        "content": f"Please provide the recipe for {recipe_name}. Include ingredients list and step-by-step cooking instructions."
                    }
                ],
                temperature=0.7,
                max_tokens=500
            )
            logger.info("API call successful")
            return response
        except openai.AuthenticationError as e:
            logger.error(f"Authentication error: {str(e)}")
            st.error("Invalid API key. Please check your OpenAI API key in the .env file.")
            raise
        except openai.RateLimitError as e:
            logger.error(f"Rate limit error: {str(e)}")
            st.error("Rate limit exceeded. Please wait a few minutes before trying again.")
            time.sleep(60)  # Wait for 60 seconds before retrying
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {str(e)}")
            st.error("Error communicating with OpenAI API. Please try again later.")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in API call: {str(e)}")
            st.error("An unexpected error occurred. Please try again later.")
            raise

    # Cache recipe data
    @st.cache_data(ttl=3600)
    def load_recipe_data():
        try:
            return pd.read_csv('final_data.csv')
        except Exception as e:
            logger.error(f"Data loading error: {str(e)}")
            st.error("Error loading recipe data")
            return None

    # Cache model
    @st.cache_resource
    def load_model(df):
        try:
            reader = Reader(rating_scale=(0, 100))
            data = Dataset.load_from_df(df[['user_id', 'recipe_code', 'ratings']], reader)
            trainset = data.build_full_trainset()
            algo = SVD()
            algo.fit(trainset)
            return algo
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            st.error("Error loading recommendation model")
            return None

    def get_chatgpt_recipe(recipe_name):
        """
        Get recipe suggestions from ChatGPT with error handling and rate limiting
        """
        try:
            logger.info(f"Generating recipe for: {recipe_name}")
            try:
                response = rate_limited_api_call(recipe_name)
            except openai.RateLimitError:
                st.warning("We're experiencing high demand. Please wait a few minutes before trying again.")
                return None
                
            if not response or not response.choices:
                logger.error("Empty response from OpenAI")
                st.error("No response received from the AI. Please try again.")
                return None
                
            recipe_text = response.choices[0].message.content
            logger.info("Received recipe text from OpenAI")
            
            # Parse response with error handling
            try:
                if "Ingredients:" not in recipe_text or "Instructions:" not in recipe_text:
                    logger.warning("Response missing required sections")
                    return {
                        "ingredients": recipe_text,
                        "instructions": "Instructions not provided in the expected format",
                        "is_ai_generated": True,
                        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                ingredients_section = recipe_text.split("Ingredients:")[1].split("Instructions:")[0].strip()
                instructions_section = recipe_text.split("Instructions:")[1].strip()
                
                return {
                    "ingredients": ingredients_section,
                    "instructions": instructions_section,
                    "is_ai_generated": True,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
            except Exception as e:
                logger.warning(f"Response parsing error: {str(e)}")
                return {
                    "ingredients": recipe_text,
                    "instructions": "Could not parse instructions",
                    "is_ai_generated": True,
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }
        except Exception as e:
            logger.error(f"Recipe generation error: {str(e)}")
            st.error("Error generating recipe suggestion. Please try again later.")
            return None

    def streamlit_app():
        # Set page configuration
        st.set_page_config(
            page_title="Recipe Search App",
            page_icon="🍳",
            layout="wide"
        )

        # Custom CSS
        st.markdown("""
            <style>
            .stApp {
                background-image: url('https://images.pexels.com/photos/775032/pexels-photo-775032.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }
            .recipe-card {
                background-color: rgba(255, 255, 255, 0.9);
                padding: 20px;
                border-radius: 10px;
                margin: 10px 0;
            }
            </style>
        """, unsafe_allow_html=True)

        # Load data and model
        df = load_recipe_data()
        if df is None:
            st.stop()
        
        algo = load_model(df)
        if algo is None:
            st.stop()

        # App header
        st.title('🍳 Recipe Search App')
        st.markdown("---")

        # Sidebar navigation
        st.sidebar.title('Navigation')
        page = st.sidebar.radio('Go to', ['Home', 'About', 'Search'])

        # Page content
        if page == 'Home':
            st.header('Welcome to Recipe Search App! 👋')
            st.write('Discover new recipes and get AI-powered cooking suggestions!')
            
            # Display some statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Recipes", len(df['recipe_name'].unique()))
            with col2:
                st.metric("Average Rating", f"{df['ratings'].mean():.1f}")
            with col3:
                st.metric("Total Reviews", len(df))

        elif page == 'About':
            st.header('About the App ℹ️')
            st.write("""
            This intelligent recipe search app combines traditional recipes with AI-powered suggestions:
            
            - 🔍 Search by recipe name or ingredients
            - 🤖 AI-generated recipes when not found in database
            - ⭐ Rating predictions for recommendations
            - 📊 Real-time recipe analysis
            """)

        elif page == 'Search':
            st.header('Recipe Search 🔍')

            # Search tabs
            tab1, tab2 = st.tabs(["Recipe Name Search", "Ingredient Search"])

            with tab1:
                recipe_name_search = st.text_input('Enter Recipe Name:')
                if st.button('Search Recipe'):
                    if recipe_name_search:
                        with st.spinner('Searching for recipe...'):
                            # Search in database
                            filtered_recipes = df[df['recipe_name'].str.contains(recipe_name_search, case=False, na=False)]

                            if not filtered_recipes.empty:
                                recipe = filtered_recipes.iloc[0]
                                with st.container():
                                    st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                                    st.subheader(f"📖 {recipe['recipe_name']}")
                                    st.markdown(f"**Ingredients:**\n{recipe['ingredients']}")
                                    st.markdown(f"**Instructions:**\n{recipe['cooking_instructions']}")
                                    st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.info("Recipe not found in database. Generating AI suggestion...")
                                recipe = get_chatgpt_recipe(recipe_name_search)
                                if recipe:
                                    with st.container():
                                        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                                        st.subheader(f"🤖 AI Generated: {recipe_name_search}")
                                        st.markdown(f"**Ingredients:**\n{recipe['ingredients']}")
                                        st.markdown(f"**Instructions:**\n{recipe['instructions']}")
                                        st.caption(f"Generated at: {recipe['generated_at']}")
                                        st.markdown('</div>', unsafe_allow_html=True)

            with tab2:
                ingredient_search = st.text_input('Enter Ingredients (comma separated):')
                if st.button('Search by Ingredients'):
                    if ingredient_search:
                        with st.spinner('Searching for recipes...'):
                            ingredients = [ingredient.strip() for ingredient in ingredient_search.split(',')]
                            filtered_recipes = df[df['ingredients'].str.contains('|'.join(ingredients), case=False, na=False)]

                            if not filtered_recipes.empty:
                                st.subheader("📚 Found Recipes")
                                for _, recipe in filtered_recipes.iterrows():
                                    with st.container():
                                        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                                        st.markdown(f"**{recipe['recipe_name']}**")
                                        st.markdown(f"**Ingredients:**\n{recipe['ingredients']}")
                                        st.markdown(f"**Instructions:**\n{recipe['cooking_instructions']}")
                                        st.markdown('</div>', unsafe_allow_html=True)
                            else:
                                st.info("No recipes found. Generating AI suggestion...")
                                recipe = get_chatgpt_recipe(f"recipe using {ingredient_search}")
                                if recipe:
                                    with st.container():
                                        st.markdown('<div class="recipe-card">', unsafe_allow_html=True)
                                        st.subheader("🤖 AI Generated Recipe Suggestion")
                                        st.markdown(f"**Ingredients:**\n{recipe['ingredients']}")
                                        st.markdown(f"**Instructions:**\n{recipe['instructions']}")
                                        st.caption(f"Generated at: {recipe['generated_at']}")
                                        st.markdown('</div>', unsafe_allow_html=True)

        # Footer
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center'>
                <p>Made with ❤️ by Your Name | Data Dining Delight</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    if __name__ == "__main__":
        streamlit_app()