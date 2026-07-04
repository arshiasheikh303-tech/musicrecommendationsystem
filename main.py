import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

songs_df = pd.read_csv("dataset.csv")


songs_df = songs_df.reset_index(drop=True)
songs_df["song_id"] = songs_df.index


songs_df["artists"] = songs_df["artists"].fillna("")
songs_df["track_genre"] = songs_df["track_genre"].fillna("")
songs_df["track_name"] = songs_df["track_name"].fillna("")


songs_df["features"] = songs_df["artists"] + " " + songs_df["track_genre"]


vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(songs_df["features"])


# CONTENT-BASED RECOMMENDER :
def recommend_by_song(song_name, songs_df, top_n=5):
    """
    Recommends songs similar to the one given, based on artist and genre.
    """
    matches = songs_df[songs_df["track_name"].str.lower() == song_name.lower()]
    if matches.empty:
        print(f"Song '{song_name}' not found.")
        return None

    idx = matches.index[0]

    song_vector = tfidf_matrix[idx]
    similarity_scores = cosine_similarity(song_vector, tfidf_matrix)[0]

    scores = list(enumerate(similarity_scores))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_indices = [i for i, score in scores[1:top_n + 1]]
    return songs_df.iloc[top_indices][["track_name", "artists", "track_genre"]]

listening_history = {
    "user_id":    [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5],
    "song_id":    [0, 1, 4, 1, 5, 0, 3, 8, 2, 6, 7, 9],
    "play_count": [5, 3, 2, 4, 5, 3, 2, 4, 5, 4, 3, 2],
}
history_df = pd.DataFrame(listening_history)


def recommend_by_user(user_id, history_df, songs_df, top_n=5):
    """
    Recommends songs based on what similar users have listened to.
    """
    user_item_matrix = history_df.pivot_table(
        index="user_id", columns="song_id", values="play_count", fill_value=0
    )

    if user_id not in user_item_matrix.index:
        print(f"User {user_id} not found.")
        return None

    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(
        user_similarity,
        index=user_item_matrix.index,
        columns=user_item_matrix.index,
    )

    similar_users = user_similarity_df[user_id].drop(user_id)
    weighted_scores = user_item_matrix.loc[similar_users.index].T.dot(similar_users)

    already_played = user_item_matrix.loc[user_id]
    weighted_scores = weighted_scores[already_played == 0]

    top_song_ids = weighted_scores.sort_values(ascending=False).head(top_n).index

    return songs_df[songs_df["song_id"].isin(top_song_ids)][
        ["song_id", "track_name", "artists", "track_genre"]
    ]



if __name__ == "__main__":
    print("=" * 60)
    print("CONTENT-BASED RECOMMENDATIONS")
    print("=" * 60)

    song_to_match = songs_df.loc[0, "track_name"]  # change this to any song in your dataset
    print(f"\nBecause you listened to '{song_to_match}':\n")

    result = recommend_by_song(song_to_match, songs_df, top_n=3)
    if result is not None:
        print(result.to_string(index=False))

    print("\n" + "=" * 60)
    print("COLLABORATIVE FILTERING RECOMMENDATIONS")
    print("=" * 60)

    target_user = 1
    print(f"\nRecommended for User {target_user}:\n")

    result2 = recommend_by_user(target_user, history_df, songs_df, top_n=3)
    if result2 is not None:
        print(result2.to_string(index=False))
