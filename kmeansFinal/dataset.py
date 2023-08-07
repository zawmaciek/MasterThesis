import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import NewType, Optional, List

import pandas as pd
from numpy import ndarray
from tqdm import tqdm

userId = NewType('userId', int)
movieId = NewType('movieId', int)
import numpy as np


@dataclass
class Movie:
    movieId: movieId
    title: str
    year: Optional[int]
    genres: list[str]
    genres_vector: list[float]
    rating: float
    rating_count: int


class Dataset:
    def __init__(self) -> None:
        self.con = sqlite3.connect("../dataset.db")
        self.cur = self.con.cursor()
        self.user_id_to_matrix_user_id: Optional[dict[userId, int]] = None
        self.matrix_user_id_to_user_id: Optional[dict[int, userId]] = None
        self.users_count = None
        self.GENRES = [
            "Adventure",
            "Animation",
            "Children",
            "Comedy",
            "Fantasy",
            "Romance",
            "Drama",
            "Action",
            "Crime",
            "Thriller",
            "Horror",
            "Mystery",
            "Sci-Fi",
            "IMAX",
            "Documentary",
            "War",
            "Musical",
            "Western",
            "Film-Noir"]
        self.movie_meta = self.get_movie_meta()
        self.all_movies = self.get_all_movies()
        self.ratings_by_user = self.get_all_ratings_grouped_by_user()
        self.label_by_user = self.get_user_label_map()

    def get_all_movies(self) -> list[Movie]:
        res = self.cur.execute("SELECT * FROM movies").fetchall()
        movies = []
        for m in tqdm(res, desc='get movies'):
            id = movieId(int(m[0]))
            raw_title = m[1]
            genres = m[2].split('|')
            if genres == ["(no genres listed)"]:
                genres = None
            regex_result = re.findall("\d\d\d\d", m[1])
            year = int(regex_result[-1]) if len(regex_result) > 0 else None
            if genres and year:
                genres_vector = []
                for genre in self.GENRES:
                    if genre in genres:
                        genres_vector.append(1.0)
                    else:
                        genres_vector.append(0)
                s = sum(genres_vector)
                genres_vector = [a / s for a in genres_vector]
                if id in self.movie_meta:
                    avg_rating, count_rating = self.movie_meta[id]
                else:
                    avg_rating, count_rating = None, None
                movies.append(Movie(id, raw_title, year, genres, genres_vector, avg_rating, count_rating))
        return movies

    def get_all_movie_ids(self) -> list[movieId]:
        res = self.cur.execute("SELECT DISTINCT movieId FROM movies").fetchall()
        return [movieId(int(a[0])) for a in res]

    def get_all_user_ids(self) -> list[userId]:
        res = self.cur.execute("SELECT DISTINCT userId FROM ratings").fetchall()
        return [userId(int(a[0])) for a in res]

    def get_all_ratings(self) -> list[tuple[userId, movieId, float]]:
        res = self.cur.execute("SELECT * FROM ratings").fetchall()
        return [(userId(int(r[0])), movieId(int(r[1])), float(r[2]) / 5.0) for r in res]

    def get_all_ratings_grouped_by_user(self) -> defaultdict[int, list[tuple[movieId, float]]]:
        ratings = self.get_all_ratings()
        ddict = defaultdict(list)
        for uid, mid, rat in tqdm(ratings, desc='ratings'):
            ddict[uid].append((mid, rat))
        return ddict

    def disconnect(self) -> None:
        self.con.close()

    def get_movie_meta(self) -> dict[movieId, tuple[float, int]]:
        res = self.cur.execute("SELECT movieId,AVG(rating),COUNT(rating) FROM ratings GROUP BY movieId")
        return {movieId(int(a[0])): (float(a[1]) / 5.0, int(a[2])) for a in res}

    def get_user_label_map(self) -> dict[userId, int]:
        res = self.cur.execute("SELECT * FROM groups")
        return {userId(int(a[0])): int(a[1]) for a in res}
