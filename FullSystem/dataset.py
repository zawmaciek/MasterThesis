import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from typing import NewType, Optional
import numpy as np
from tqdm import tqdm

userId = NewType('userId', int)
movieId = NewType('movieId', int)
tagId = NewType('tagId', int)


@dataclass
class User:
    id: userId
    vector: list[float]
    rank: float


@dataclass
class TaggedMovie:
    movieId: movieId
    tags: list[float]
    title: str
    rank: float = 0.0


@dataclass
class Movie:
    movieId: movieId
    title: str
    year: Optional[int]
    genres: list[str]
    genres_vector: list[float]


class Dataset:
    def __init__(self) -> None:
        self.movie_id_title_mapping = None
        self.movie_vector_mapping = None
        self.tag_id_to_matrix_tag_id = None
        self.matrix_tag_id_to_tag_id = None
        self.movie_id_to_matrix_movie_id = None
        self.matrix_movie_id_to_movie_id = None
        self.user_id_to_matrix_user_id = None
        self.matrix_user_id_to_user_id = None
        self.tags_count = None
        self.users_count = None
        self.movies_count = None
        self.label_by_user = None
        print("LOADING DATASET")
        self.con = sqlite3.connect("../dataset.db")
        self.cur = self.con.cursor()
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
        self.all_movies = self.get_all_movies()
        self.all_movie_ids = self.get_all_movie_ids()
        self.ratings_by_user = self.get_all_ratings_grouped_by_user()
        self.set_mappings()
        print("DATASET LOADED")

    def set_mappings(self):
        self.all_movie_ids = self.get_all_movie_ids()
        self.all_user_ids = self.get_all_user_ids()
        self.all_tag_ids = self.get_all_tag_ids()
        self.movies_count = len(self.all_movie_ids)
        self.users_count = len(self.all_user_ids)
        self.tags_count = len(self.all_tag_ids)
        self.matrix_user_id_to_user_id = dict()
        self.user_id_to_matrix_user_id = dict()
        self.matrix_movie_id_to_movie_id = dict()
        self.movie_id_to_matrix_movie_id = dict()
        self.matrix_tag_id_to_tag_id = dict()
        self.tag_id_to_matrix_tag_id = dict()
        for i, val in tqdm(enumerate(self.all_movie_ids), desc='movie mappings'):
            self.matrix_movie_id_to_movie_id[i] = val
            self.movie_id_to_matrix_movie_id[val] = i
        for i, val in tqdm(enumerate(self.all_user_ids), desc='user mappings'):
            self.matrix_user_id_to_user_id[i] = val
            self.user_id_to_matrix_user_id[val] = i
        for i, val in tqdm(enumerate(self.all_tag_ids), desc='tag mappings'):
            self.matrix_tag_id_to_tag_id[i] = val
            self.tag_id_to_matrix_tag_id[val] = i
        self.movie_vector_mapping = self.get_movies_vectors_mapping()
        self.movie_id_title_mapping = self.get_movie_id_title_mapping()

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
                genres_vector = []
                for genre in self.GENRES:
                    if genre in genres:
                        genres_vector.append(1.0)
                    else:
                        genres_vector.append(0)
                s = sum(genres_vector)
                genres_vector = [a / s for a in genres_vector]
                movies.append(Movie(movieId(id), raw_title, year, genres, genres_vector))
        return movies

    def get_all_movie_ids(self) -> set[movieId]:
        return {a.movieId for a in self.all_movies}

    def get_all_user_ids(self) -> list[userId]:
        res = self.cur.execute("SELECT DISTINCT userId FROM ratings").fetchall()
        return [userId(int(a[0])) for a in res]

    def get_all_tag_ids(self) -> list[tagId]:
        res = self.cur.execute('SELECT DISTINCT tagId FROM "genome-tags"').fetchall()
        return [tagId(int(a[0])) for a in res]

    def get_all_ratings(self) -> list[tuple[userId, movieId, float]]:
        res = self.cur.execute("SELECT * FROM ratings").fetchall()
        return [(userId(int(r[0])), movieId(int(r[1])), float(r[2]) / 5.0) for r in res if int(r[1]) in self.all_movie_ids]

    def get_all_ratings_grouped_by_user(self) -> defaultdict[int, list[tuple[movieId, float]]]:
        ratings = self.get_all_ratings()
        ddict = defaultdict(list)
        for uid, mid, rat in tqdm(ratings, desc='ratings'):
            ddict[uid].append((mid, rat))
        return ddict

    def disconnect(self) -> None:
        self.con.close()

    def get_reduced_matrix(self):
        height = self.users_count
        matrix = np.zeros((height, len(self.GENRES)))
        movies_map = {a.movieId: a for a in self.all_movies}
        ratings_by_user = self.ratings_by_user
        for user in tqdm(ratings_by_user, desc="get reduced matrix"):
            user_vector = [0 for _ in range(len(self.GENRES))]
            for m_id, rating in ratings_by_user[user]:
                if m_id in movies_map:
                    movie_vector = movies_map[m_id].genres_vector
                    user_vector = [user_vector[i] + movie_vector[i] * rating for i in range(len(self.GENRES))]
            s = sum(user_vector)
            user_vector = [i / s for i in user_vector]
            for i in range(len(self.GENRES)):
                matrix[self.user_id_to_matrix_user_id[user], i] = user_vector[i]
        return matrix

    def get_reduced_matrix_for_user(self, ratings: list[tuple[movieId, float]]) -> list[float]:
        movies_map = {a.movieId: a for a in self.all_movies}
        user_vector = [0 for _ in range(len(self.GENRES))]
        for m_id, rating in ratings:
            if m_id in movies_map:
                movie_vector = movies_map[m_id].genres_vector
                user_vector = [user_vector[i] + movie_vector[i] * rating for i in range(len(self.GENRES))]
        s = sum(user_vector)
        user_vector = [i / s for i in user_vector]
        return user_vector

    def get_movie_ids_from_users(self, users: list[userId], ignore: list[movieId] = None) -> list[movieId]:
        if ignore is None:
            ignore = []
        ratings = self.ratings_by_user
        ddict = defaultdict(lambda: 0.0)
        movies_map = {a.movieId: a for a in self.all_movies}
        for user in users:
            for m_id, rating in ratings[user]:
                ddict[m_id] += rating
        top = sorted(ddict.items(), key=lambda item: item[1], reverse=True)
        return [movies_map[a[0]].movieId for a in top if movies_map[a[0]].movieId not in ignore][0:10]

    @cache
    def get_users_full_vectors(self, label: int) -> list[User]:
        users = []
        length = self.movies_count
        for user in tqdm(self.all_user_ids, desc="get user vectors"):
            if self.label_by_user[user] == label:
                vector = [0 for _ in range(length)]
                for m_id, rating in self.ratings_by_user[user]:
                    vector[self.movie_id_to_matrix_movie_id[m_id]] = rating
                users.append(User(user, vector, 0.0))
        return users

    def get_full_user_vector_from_ratings(self, ratings: list[tuple[movieId, float]]) -> list[float]:
        vector = [0 for _ in range(self.movies_count)]
        for m_id, rating in ratings:
            vector[self.movie_id_to_matrix_movie_id[m_id]] = rating
        return vector

    def get_movies_vectors(self) -> list[TaggedMovie]:
        all = []
        for m_id in self.movie_vector_mapping:
            all.append(TaggedMovie(m_id, self.movie_vector_mapping[m_id], self.movie_id_title_mapping[m_id]))
        return all

    def get_movies_vectors_mapping(self) -> defaultdict[movieId, list[float]]:
        d = defaultdict(lambda: [0 for _ in range(self.tags_count)])
        res = self.cur.execute('SELECT * FROM "genome-scores"').fetchall()
        for row in tqdm(res, desc="get tags"):
            m_id = movieId(int(row[0]))
            if m_id in self.all_movie_ids:
                t_id = tagId(int(row[1]))
                t_matrix_id = self.tag_id_to_matrix_tag_id[t_id]
                score = float(row[2])
                d[m_id][t_matrix_id] = score
        return d

    def get_movie_id_title_mapping(self) -> dict[movieId, str]:
        return {movie.movieId: movie.title for movie in self.all_movies}
