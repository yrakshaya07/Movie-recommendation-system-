import streamlit as st
import pandas as pd
from recommendation import get_content_recommendations, get_collaborative_recommendations


try:
    movies = pd.read_csv("D:/project/movies_metadata.csv", encoding="utf-8")
except Exception as e:
    st.error(f"Failed to load movies_metadata.csv: {e}")
    st.stop()

movie_titles = movies['title'].dropna().unique()
movie_titles.sort()

st.title("🎬 Movie Recommendation System")
st.markdown("Get movie recommendations using *ML-based filtering techniques*.")


selected_movie = st.selectbox("Choose a movie you like:", movie_titles)


rec_type = st.radio("Choose recommendation method:", ["Content-Based", "Collaborative Filtering"])

if st.button("Recommend"):
    st.subheader("Top 5 Recommendations 🎉")

    try:
        if rec_type == "Content-Based":
            recs = get_content_recommendations(selected_movie)
        else:
            recs = get_collaborative_recommendations(selected_movie)

        if not recs or (isinstance(recs, list) and "not" in recs[0].lower()):
            st.warning(recs[0])
        else:
            for i, movie in enumerate(recs, start=1):
                st.write(f"{i}. {movie}")
    except Exception as e:
        st.error(f"⚠️ An error occurred while generating recommendations:\n\n{e}")
