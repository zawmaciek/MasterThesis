import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import NewType, Optional

from numpy import ndarray
from tqdm import tqdm

userId = NewType('userId', int)
movieId = NewType('movieId', int)
import numpy as np


@dataclass
class Movie:
    movieId: movieId
    title: str
    genres: list[str]


@dataclass
class Movie:
    movieId: movieId
    title: str
    year: Optional[int]
    genres: list[str]


class Dataset:
    def __init__(self) -> None:
        self.con = sqlite3.connect("dataset.db")
        self.cur = self.con.cursor()
        self.movie_id_to_matrix_movie_id = None
        self.matrix_movie_id_to_movie_id = None
        self.user_id_to_matrix_user_id = None
        self.matrix_user_id_to_user_id = None
        self.users_count = None
        self.movies_count = None
        self.set_mappings()

    def set_mappings(self):
        all_movie_ids = self.get_all_movie_ids()
        all_user_ids = self.get_all_user_ids()
        self.movies_count = len(all_movie_ids)
        self.users_count = len(all_user_ids)
        self.matrix_user_id_to_user_id = dict()
        self.user_id_to_matrix_user_id = dict()
        self.matrix_movie_id_to_movie_id = dict()
        self.movie_id_to_matrix_movie_id = dict()
        for i, val in tqdm(enumerate(all_movie_ids), desc='movie mappings'):
            self.matrix_movie_id_to_movie_id[i] = val
            self.movie_id_to_matrix_movie_id[val] = i
        for i, val in tqdm(enumerate(all_user_ids), desc='user mappings'):
            self.matrix_user_id_to_user_id[i] = val
            self.user_id_to_matrix_user_id[val] = i

    def get_all_movies(self) -> list[Movie]:
        res = self.cur.execute("SELECT * FROM movies").fetchall()
        movies = []
        for m in tqdm(res, desc='get movies'):
            id = int(m[0]) - 1
            raw_title = m[1]
            genres = m[2].split('|')
            if genres == ["(no genres listed)"]:
                genres = None
            regex_result = re.findall("\d\d\d\d", m[1])
            year = int(regex_result[-1]) if len(regex_result) > 0 else None
            if genres and year:
                movies.append(Movie(movieId(id), raw_title, year, genres))
        return movies

    def get_all_movie_ids(self) -> list[movieId]:
        res = self.cur.execute("SELECT DISTINCT movieId FROM movies").fetchall()
        return [movieId(int(a[0]) - 1) for a in res]

    def get_max_movie_id(self) -> movieId:
        return max(self.get_all_movie_ids())

    def get_all_user_ids(self) -> list[userId]:
        res = self.cur.execute("SELECT DISTINCT userId FROM ratings").fetchall()
        return [userId(int(a[0]) - 1) for a in res]

    def get_max_user_id(self) -> userId:
        return max(self.get_all_user_ids())

    def get_all_ratings(self) -> list[tuple[userId, movieId, float]]:
        res = self.cur.execute("SELECT * FROM ratings").fetchall()
        return [(userId(int(r[0]) - 1), movieId(int(r[1]) - 1), float(r[2]) / 5.0) for r in res]

    def get_all_ratings_grouped_by_user(self) -> defaultdict[int, list[tuple[movieId, float]]]:
        ratings = self.get_all_ratings()
        ddict = defaultdict(list)
        for uid, mid, rat in tqdm(ratings, desc='ratings'):
            ddict[uid].append((mid, rat))
        return ddict

    def disconnect(self) -> None:
        self.con.close()

    def get_matrix(self):
        length = self.movies_count
        height = self.users_count
        matrix = np.zeros((height, length))
        for uid, mid, rating in tqdm(self.get_all_ratings(), desc='get matrix'):
            matrix[self.user_id_to_matrix_user_id[uid], self.movie_id_to_matrix_movie_id[mid]] = rating
        return matrix
