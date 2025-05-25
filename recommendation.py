import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


try:
    movies = pd.read_csv("D:/project/movies_metadata.csv", encoding="utf-8")
    ratings = pd.read_csv("D:/project/ratings.csv", encoding="utf-8")

    
    movies = movies[movies['id'].apply(lambda x: str(x).isdigit())]
    movies['movieId'] = movies['id'].astype(int)
except Exception as e:
    raise FileNotFoundError(f"Error loading data files: {e}")

def get_content_recommendations(title, top_n=5):
    if 'genres' not in movies.columns or 'title' not in movies.columns:
        return ["Movies data missing required columns."]

    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['genres'].fillna(''))

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

    if title not in indices:
        return ["Movie not found in dataset."]

    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices].tolist()

def get_collaborative_recommendations(title, top_n=5):
    if 'movieId' not in movies.columns or 'movieId' not in ratings.columns:
        return ["Missing 'movieId' in movies or ratings data."]

    if 'rating' not in ratings.columns or 'userId' not in ratings.columns:
        return ["Missing 'userId' or 'rating' in ratings data."]

    try:
        user_movie_matrix = ratings.merge(movies[['movieId', 'title']], on='movieId')
    except Exception:
        return ["Error merging ratings and movies data."]

    matrix = user_movie_matrix.pivot_table(index='userId', columns='title', values='rating').fillna(0)

    if title not in matrix.columns:
        return ["Movie not rated by users. Try another."]

    try:
        model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
        model_knn.fit(matrix.T.values)

        query_index = list(matrix.columns).index(title)
        distances, indices = model_knn.kneighbors([matrix.T.values[query_index]], n_neighbors=top_n+1)

        recommended = [matrix.columns[i] for i in indices.flatten()][1:]
        return recommended
    except Exception as e:
        return [f"Collaborative filtering failed: {e}"]
