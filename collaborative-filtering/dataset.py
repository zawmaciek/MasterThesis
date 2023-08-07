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
class User:
    id: userId
    vector: list[float]
    rank: float


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
        self.con = sqlite3.connect("../dataset.db")
        self.cur = self.con.cursor()
        self.label_by_user = self.get_user_label_map()
        self.all_movies = self.get_all_movies()
        self.all_movie_ids = self.get_all_movie_ids()
        self.ratings_by_user = self.get_all_ratings_grouped_by_user()
        self.set_mappings()

    def set_mappings(self):
        all_movie_ids = self.all_movie_ids
        all_user_ids = self.get_all_user_ids()
        self.movies_count = len(all_movie_ids)
        self.users_count = len(all_user_ids)
        self.matrix_movie_id_to_movie_id = dict()
        self.movie_id_to_matrix_movie_id = dict()
        for i, val in tqdm(enumerate(all_movie_ids), desc='movie mappings'):
            self.matrix_movie_id_to_movie_id[i] = val
            self.movie_id_to_matrix_movie_id[val] = i

    def get_all_movies(self) -> list[Movie]:
        res = self.cur.execute("SELECT * FROM movies").fetchall()
        movies = []
        for m in tqdm(res, desc='get movies'):
            id = int(m[0])
            raw_title = m[1]
            genres = m[2].split('|')
            if genres == ["(no genres listed)"]:
                genres = None
            regex_result = re.findall("\d\d\d\d", m[1])
            year = int(regex_result[-1]) if len(regex_result) > 0 else None
            if genres and year:
                movies.append(Movie(movieId(id), raw_title, year, genres))
        return movies

    def get_all_movie_ids(self) -> set[movieId]:
        return {a.movieId for a in self.all_movies}

    def get_all_user_ids(self) -> list[userId]:
        res = self.cur.execute("SELECT DISTINCT userId FROM ratings").fetchall()
        return [userId(int(a[0])) for a in res]

    def get_all_ratings(self) -> list[tuple[userId, movieId, float]]:
        res = self.cur.execute("SELECT * FROM ratings").fetchall()
        return [(userId(int(r[0])), movieId(int(r[1])), float(r[2]) / 5.0) for r in res if int(r[1]) in self.all_movie_ids]

    def get_all_ratings_per_label(self, label: int) -> list[tuple[userId, movieId, float]]:
        def get_users_per_label(label: int) -> list[userId]:
            res = self.cur.execute(f'SELECT userId FROM groups WHERE "group"=={label}')
            return [userId(int(a[0])) for a in res]

        users = get_users_per_label(label)
        return [a for a in self.get_all_ratings() if a[0] in users]

    def get_all_ratings_grouped_by_user(self) -> defaultdict[int, list[tuple[movieId, float]]]:
        ratings = self.get_all_ratings()
        ddict = defaultdict(list)
        for uid, mid, rat in tqdm(ratings, desc='ratings'):
            ddict[uid].append((mid, rat))
        return ddict

    def disconnect(self) -> None:
        self.con.close()

    def get_user_label_map(self) -> dict[userId, int]:
        res = self.cur.execute("SELECT * FROM groups")
        return {userId(int(a[0])): int(a[1]) for a in res}

    def get_user_vectors(self, label: int) -> list[User]:
        users = []
        length = self.movies_count
        for user in tqdm(self.get_all_user_ids(), desc="get user vectors"):
            if self.label_by_user[user] == label:
                vector = [0 for _ in range(length)]
                for m_id, rating in self.ratings_by_user[user]:
                    vector[self.movie_id_to_matrix_movie_id[m_id]] = rating
                users.append(User(user, vector, 0.0))
        return users
