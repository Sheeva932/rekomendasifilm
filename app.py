
import streamlit as st
import joblib
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load model dan data
df_all = joblib.load('df_all.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Fungsi rekomendasi film
def recommend_film(title, num_recommendations=6):
    title = title.lower()
    matches = df_all[df_all['title'].str.lower().str.contains(title, na=False)]

    if matches.empty:
        return f"Film dengan judul '{title}' tidak ditemukan."

    idx = matches.index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]

    film_indices = [i[0] for i in sim_scores]
    similarities = [i[1] for i in sim_scores]

    result = df_all[['title', 'genres', 'director', 'cast', 'poster_url']].iloc[film_indices].copy()
    result['cosine_similarity'] = similarities

    return result

# Tampilan Streamlit
st.set_page_config(page_title="Sistem Rekomendasi Film", layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>🎬 Sistem Rekomendasi Film</h1>", unsafe_allow_html=True)
st.markdown("## Masukkan Judul film yang kamu suka :")

col_input, col_btn = st.columns([5, 1])
with col_input:
    input_title = st.text_input("", placeholder="Contoh: John Wick")
with col_btn:
    cari = st.button("Cari Rekomendasi")

if cari and input_title:
    hasil = recommend_film(input_title)
    
    if isinstance(hasil, str):
        st.warning(hasil)
    else:
        st.markdown("### Berikut hasil rekomendasi film mu :")
        cols = st.columns(3)
        for i, (_, row) in enumerate(hasil.iterrows()):
            with cols[i % 3]:
                st.image(row['poster_url'], width=180)
                st.markdown(f"""
                    <div style='background-color: #f0f0f0; padding: 10px; border-radius: 15px; margin-top: 10px'>
                    <b>Title</b>: {row['title']}<br>
                    <b>Genre</b>: {row['genres']}<br>
                    <b>Director</b>: {row['director']}<br>
                    <b>Cast</b>: {row['cast']}
                    </div>
                """, unsafe_allow_html=True)
