
import streamlit as st
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load file .pkl
df_all = joblib.load('df_all.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')
tfidf_matrix = joblib.load('tfidf_matrix.pkl')

cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

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

    result = df_all.iloc[film_indices].copy()
    result['cosine_similarity'] = similarities
    return result

# Tampilan utama
st.set_page_config(layout="wide")
st.markdown("<h1 style='text-align: center; color: white;'>🎬 Sistem Rekomendasi Film</h1>", unsafe_allow_html=True)

# Input besar
input_title = st.text_input(
    "",
    placeholder="Masukkan Judul Film Favoritmu...",
    key="film_input"
)

st.markdown("""
<style>
input[data-baseweb="input"] {
    height: 50px;
    font-size: 18px;
}
.stButton>button {
    padding: 0.75em 2em;
    font-size: 16px;
    border-radius: 10px;
}
.stContainer {
    max-width: 100%;
}
</style>
""", unsafe_allow_html=True)

if st.button("Cari Rekomendasi"):
    hasil = recommend_film(input_title)

    if isinstance(hasil, str):
        st.warning(hasil)
    else:
        st.markdown("<h3 style='text-align: center; color: white;'>Berikut hasil rekomendasi film mu :</h3>", unsafe_allow_html=True)
        
        # 2 baris isi 3 kolom
        for i in range(0, len(hasil), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(hasil):
                    film = hasil.iloc[i + j]
                    with cols[j]:
                        st.image(f"https://via.placeholder.com/300x450.png?text={film['title'].replace(' ', '+')}", use_column_width=True)
                        st.markdown(f"""
                            <div style='background-color: white; padding: 15px; border-radius: 15px; margin-top: 10px;'>
                                <strong>Title:</strong> {film['title']}<br>
                                <strong>Genre:</strong> {film['genres']}<br>
                                <strong>Director:</strong> {film['director']}<br>
                                <strong>Cast:</strong> {film['cast']}
                            </div>
                        """, unsafe_allow_html=True)
