import streamlit as st
from model import recommend , new_df

st.title("🎬 Movie Recommendation System")

movie_name = st.selectbox("Select a movie", new_df['title'].values)

num_movies = st.slider("Number of recommendations", 5, 20, 10)

if st.button("Recommend"):

    results = recommend(movie_name, num_movies)

    for i, movie in enumerate(results, start=1):
        st.write(f"{i}. {movie}")