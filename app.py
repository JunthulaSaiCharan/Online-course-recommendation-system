import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import linear_kernel

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Online Course Recommendation System",
    layout="wide"
)

st.title(" Online Course Recommendation System")
st.markdown("Hybrid Recommendation using **SVD + Item-CF + Content-Based Filtering**")

# -------------------------------
# Load Data & Models
# -------------------------------
@st.cache_resource
def load_models():
    hybrid_data = joblib.load("best_hybrid_model.pkl")
    svd_model = hybrid_data["svd"]
    item_cf_model = hybrid_data["item_cf"]
    tfidf_matrix = hybrid_data["tfidf_matrix"]
    course_idx_map = hybrid_data["course_idx_map"]
    df = hybrid_data["df"]
    return svd_model, item_cf_model, tfidf_matrix, course_idx_map, df

svd_algo, item_cf_algo, tfidf_matrix, course_idx_map, df = load_models()

# -------------------------------
# Helper Functions
# -------------------------------
def recommend_knn_item(user_id, top_n=10):
    try:
        inner_uid = item_cf_algo.trainset.to_inner_uid(user_id)
    except ValueError:
        return []

    user_rated = [iid for (iid, _) in item_cf_algo.trainset.ur[inner_uid]]

    predictions = []
    for inner_iid in item_cf_algo.trainset.all_items():
        if inner_iid in user_rated:
            continue
        raw_iid = item_cf_algo.trainset.to_raw_iid(inner_iid)
        est = item_cf_algo.predict(user_id, raw_iid).est
        predictions.append((raw_iid, est))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_n]


def recommend_content_based(course_id, top_n=10):
    if course_id not in course_idx_map:
        return []

    idx = course_idx_map[course_id]
    sim_scores = linear_kernel(
        tfidf_matrix[idx], tfidf_matrix
    ).flatten()

    similar_indices = sim_scores.argsort()[::-1]
    similar_indices = [i for i in similar_indices if i != idx][:top_n]

    return [(df.iloc[i]["course_id"], sim_scores[i]) for i in similar_indices]


def hybrid_recommend(user_id, top_n=10):
    # SVD
    svd_preds = []
    for cid in df["course_id"].unique():
        if not ((df["user_id"] == user_id) & (df["course_id"] == cid)).any():
            svd_preds.append((cid, svd_algo.predict(user_id, cid).est))

    svd_preds = sorted(svd_preds, key=lambda x: x[1], reverse=True)[:top_n]
    svd_ids = [c for c, _ in svd_preds]

    st.markdown("### 🔍 Why this course was recommended")
    plot_score_breakdown(hybrid_ids[0], user_id)
    



    # Item-CF
    item_ids = [c for c, _ in recommend_knn_item(user_id, top_n)]

    # Content
    content_ids = []
    for cid in svd_ids[:3]:
        content_ids.extend([c for c, _ in recommend_content_based(cid, 3)])

    final_ids = list(dict.fromkeys(svd_ids + item_ids + content_ids))
    return final_ids[:top_n]


# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("User Input")

user_ids = df["user_id"].unique().tolist()
user_id = st.sidebar.selectbox("Select User ID", user_ids)
top_n = st.sidebar.slider("Number of Recommendations", 5, 20, 10)

# -------------------------------
# Recommendation Output
# -------------------------------
if st.sidebar.button("Get Recommendations"):

    st.subheader(f"Recommendations for User ID: {user_id}")

    col1, col2 = st.columns(2)

    # ---- SVD ----
    with col1:
        st.markdown("### SVD Recommendations")
        svd_recs = sorted(
            [(cid, svd_algo.predict(user_id, cid).est)
             for cid in df["course_id"].unique()
             if not ((df["user_id"] == user_id) & (df["course_id"] == cid)).any()],
            key=lambda x: x[1], reverse=True
        )[:top_n]

        for cid, score in svd_recs:
            name = df[df["course_id"] == cid]["course_name"].values[0]
            st.write(f"**{name}** — {round(score, 2)}")

    # ---- Item CF ----
    with col2:
        st.markdown("### Item-Based CF")
        item_recs = recommend_knn_item(user_id, top_n)
        for cid, score in item_recs:
            name = df[df["course_id"] == cid]["course_name"].values[0]
            st.write(f"**{name}** — {round(score, 2)}")

    # ---- Content ----
    st.markdown("### Content-Based Recommendations")
    ref_course = svd_recs[0][0]
    content_recs = recommend_content_based(ref_course, top_n)

    for cid, sim in content_recs:
        name = df[df["course_id"] == cid]["course_name"].values[0]
        st.write(f"**{name}** — Similarity: {round(sim, 3)}")

    # ---- Hybrid ----
    st.markdown("### Hybrid Recommendations (Final)")
    hybrid_ids = hybrid_recommend(user_id, top_n)

    for cid in hybrid_ids:
        name = df[df["course_id"] == cid]["course_name"].values[0]
        st.success(name)

import matplotlib.pyplot as plt

def plot_score_breakdown(course_id, user_id):
    svd_score = svd_algo.predict(user_id, course_id).est

    item_score = item_cf_algo.predict(user_id, course_id).est

    cert_boost = df[df['course_id']==course_id]['certification_offered'].iloc[0] * 0.1

    scores = {
        "SVD": svd_score,
        "Item-CF": item_score,
        "Certification Boost": cert_boost
    }

    plt.figure()
    plt.bar(scores.keys(), scores.values())
    plt.title("Score Contribution Breakdown")
    plt.ylabel("Score Value")
    st.pyplot(plt)

def plot_user_ratings(user_id):
    user_data = df[df['user_id'] == user_id]

    plt.figure()
    plt.hist(user_data['rating'])
    plt.title("User Engagement Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    st.pyplot(plt)

def plot_content_similarity(course_id):
    idx = course_idx_map[course_id]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()

    top_scores = sorted(sim_scores, reverse=True)[1:6]

    plt.figure()
    plt.bar(range(len(top_scores)), top_scores)
    plt.title("Top Content Similarity Scores")
    plt.xlabel("Similar Courses")
    plt.ylabel("Cosine Similarity")
    st.pyplot(plt)

def plot_model_contribution():
    labels = ['SVD', 'Item-CF', 'Content']
    sizes = [40, 35, 25]

    plt.figure()
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.title("Hybrid Model Contribution")
    st.pyplot(plt)


# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("Developed with using Streamlit")
