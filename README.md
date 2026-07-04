# musicrecommendationsystem
Simple content-based and collaborative filtering music recommender built with pandas and scikit-learn.
# Music Recommendation System

A simple Python music recommender that suggests songs using two different approaches:

- **Content-Based Filtering** — recommends songs similar to a given track, based on artist and genre, using TF-IDF and cosine similarity.
- **Collaborative Filtering** — recommends songs based on what similar users have listened to, using a user-song play count matrix.

Built with `pandas` and `scikit-learn` on a dataset of 114,000 tracks with metadata such as artist, genre, and popularity.

## Features

- Find songs similar to any track in the dataset by artist and genre
- Recommend new songs to a user based on listening history from similar users
- Lightweight, dependency-minimal, and easy to read/modify
- Handles large datasets efficiently (avoids building a full similarity matrix for the whole catalog)

## Dataset

This project expects a `dataset.csv` file with (at minimum) the following columns:

| Column        | Description                          |
|---------------|---------------------------------------|
| `track_name`  | Name of the song                      |
| `artists`     | Artist(s) of the song                 |
| `track_genre` | Genre of the song                     |
| `popularity`  | Popularity score (used for ranking)   |

> A Kaggle-style Spotify tracks dataset works well here — feel free to swap in your own as long as the columns are named the same or updated in the code.

## Installation

```bash
git clone https://github.com/your-username/music-recommendation.git
cd music-recommendation
pip install pandas scikit-learn
```

Place your `dataset.csv` file in the project's root folder.

## Usage

Run the script directly:

```bash
python main.py
```

By default, it will:
1. Print content-based recommendations for the first song in the dataset
2. Print collaborative filtering recommendations for a sample user

To try it with your own song, change this line in `main.py`:

```python
song_to_match = songs_df.loc[0, "track_name"]
```

to any song title in your dataset, for example:

```python
song_to_match = "Comedy"
```

To try a different user for collaborative filtering, change:

```python
target_user = 1
```

## Example Output

```
============================================================
CONTENT-BASED RECOMMENDATIONS
============================================================

Because you listened to 'Comedy':

                          track_name     artists track_genre
                                 Koi Gen Hoshino    acoustic
I Wanna Be Your Ghost (feat. Ghosts) Gen Hoshino    acoustic
                             FUSHIGI Gen Hoshino    acoustic

============================================================
COLLABORATIVE FILTERING RECOMMENDATIONS
============================================================

Recommended for User 1:

 song_id                 track_name                   artists track_genre
       3 Can't Help Falling In Love              Kina Grannis    acoustic
       5       Days I Will Remember              Tyrone Wells    acoustic
       8                      Lucky Jason Mraz;Colbie Caillat    acoustic
```

## How It Works

**Content-based recommender**
1. Combines each song's artist and genre into a single text field.
2. Converts that text into TF-IDF vectors.
3. Compares the chosen song's vector against all others using cosine similarity.
4. Returns the most similar songs, ranked by similarity score.

**Collaborative filtering recommender**
1. Builds a user-song matrix from play counts.
2. Finds users with similar listening patterns using cosine similarity.
3. Recommends songs that similar users played but the target user hasn't heard yet.

## Notes

- The listening history used for collaborative filtering in this project is sample data for demonstration purposes. Replace it with real user data for production use.
- Recommendations are only as good as the dataset — genre and artist labels should be consistent for best results.

## License

This project is open source and available under the MIT License.
