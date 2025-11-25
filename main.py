import pandas as pd
import numpy as np
import json
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

movies_df = pd.read_csv("movies.csv")
credits_df = pd.read_csv("credits.csv")


# REQUIRED COLUMNS
# id
# title
# genres
# keywords
# overview
# production_companies
# production_countries
# release_data

movies_df = movies_df[['id','title','genres','keywords','overview','production_companies','production_countries','release_date']]


def get_genres(obj):
    l = []
    for i in json.loads(obj):
        l.append(i['name'])
    return l
        

movies_df['genres'] = movies_df['genres'].apply(get_genres)

movies_df['keywords'] = movies_df['keywords'].apply(get_genres)

movies = pd.merge(movies_df,credits_df,on="title")

movies = movies.drop(columns=['movie_id'])

movies["overview"] = movies["overview"].astype(str).apply(lambda x: x.split())

movies['production_companies'] = movies['production_companies'].apply(get_genres)

movies['production_countries'] = movies['production_countries'].apply(get_genres)

movies["release_date"] = movies["release_date"].astype(str).apply(lambda x: x.split())

def get_cast(obj):
    l = []
    count = 0
    for i in json.loads(obj):
        if count < 3:
            l.append(i['name'])
            count = count + 1
        else:
            break
    return l
        
movies['cast'] = movies['cast'].apply(get_cast)


def get_director(obj):
    l = []
    for i in json.loads(obj):
        if i['job'] == "Director":
            l.append(i['name'])
            break
        
    return l

movies['crew'] = movies['crew'].apply(get_director)

def clean_list_columns(df, columns):
    """
    Cleans list-type string columns.
    
    For each column in `columns`, every list item will be:
        - converted to lowercase
        - stripped of leading/trailing spaces
        - spaces removed inside (bruce wayne -> brucewayne)

    Args:
        df (pd.DataFrame): Your DataFrame
        columns (list): List of column names to clean

    Returns:
        pd.DataFrame: Updated DataFrame with cleaned lists
    """

    for col in columns:
        df[col] = df[col].apply(
            lambda lst: [item.lower().replace(" ", "").strip() for item in lst]
            if isinstance(lst, list) else lst
        )
    return df

columns_to_clean = ["genres", "keywords", "overview", "production_companies","production_countries","cast","crew"]

movies = clean_list_columns(movies, columns_to_clean)

movies['tags'] = movies.apply(lambda x: x["genres"] + x["keywords"] + x["overview"] + x["production_companies"] + x["production_countries"] + x["release_date"]+ x["cast"]+ x["crew"], axis=1)

movies = movies.drop(columns=['genres','keywords','overview','production_companies','production_countries','release_date','cast','crew'])

movies["tags"] = movies["tags"].apply(lambda x: [i.lower() for i in x])
movies["tags"] = movies["tags"].apply(lambda x: " ".join(x))


ps = PorterStemmer()
def stem_text(text):
    return " ".join([ps.stem(word) for word in text.split()])

movies["tags"] = movies["tags"].apply(stem_text)


cv = CountVectorizer(
    max_features=5000,      # use top 5000 words
    stop_words='english'    # removes 'in', 'from', 'the', 'and', etc.
)

vectors = cv.fit_transform(movies["tags"]).toarray()
similarity = cosine_similarity(vectors)



def recommend(movie_title, top_n=10):
    # Step 1: Check if movie exists
    if movie_title not in movies["title"].values:
        return f"Movie '{movie_title}' not found in database."

    # Step 2: Get index of this movie in DataFrame
    movie_index = movies[movies["title"] == movie_title].index[0]

    # Step 3: Get similarity scores for this movie
    distances = similarity[movie_index]  # similarity[] returned from cosine_similarity(vectors)

    # Step 4: Enumerate & sort in descending order
    sorted_list = sorted(
        list(enumerate(distances)),
        key=lambda x: x[1],
        reverse=True
    )

    # Step 5: Remove the first entry (movie itself)
    sorted_list = sorted_list[1:]

    # Step 6: Pick top N movies
    recommended_movies = []
    for idx, score in sorted_list[:top_n]:
        movie_id = movies.iloc[idx].id
        movie_name = movies.iloc[idx].title
        recommended_movies.append({
            "id": int(movie_id),
            "title": movie_name,
            "similarity": float(score)
        })

    return recommended_movies


app = FastAPI()

# Allow React / any frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Movie recommendation API is running!"}

@app.get("/recommend")
def recommend_api(title: str, top_n: int = 10):
    """
    Example: /recommend?title=Avatar&top_n=5
    """
    result = recommend(title, top_n=top_n)
    return {"movie": title, "recommendations": result }