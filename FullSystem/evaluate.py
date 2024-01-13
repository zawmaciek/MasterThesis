import random
import matplotlib.pyplot as plt
from tqdm import tqdm

from system import System
from dataset import Dataset

import seaborn as sns


def accuracy_1(s):
    d = s.dataset
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 10000)
    jaccard_indexes = []
    t_users = tqdm(test_users)
    for user in t_users:
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        number_of_movies = 20
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:number_of_movies]
        if len(top_movies) == number_of_movies:
            half_number_of_movies = number_of_movies // 2
            unknown_liked_movies = {a[0] for a in top_movies[:half_number_of_movies]}
            known_liked_movies = top_movies[half_number_of_movies:]
            recs, _ = s.kmeans.get_reccomendations_for_user(known_liked_movies, half_number_of_movies)
            recommendations_set = set(recs)
            jaccard_index = len(unknown_liked_movies.intersection(recommendations_set)) / len(unknown_liked_movies.union(recommendations_set))
            jaccard_indexes.append(jaccard_index)
            if jaccard_indexes:
                t_users.set_description(f"Jaccard eval: {sum(jaccard_indexes) / len(jaccard_indexes)}")
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Indeks Jaccarda', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.savefig('accuracy_1.png')
    plt.clf()


def accuracy_2(s):
    d = s.dataset
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 10000)
    jaccard_indexes = []
    t_users = tqdm(test_users)
    for user in t_users:
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        number_of_movies = 20
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:number_of_movies]
        if len(top_movies) == number_of_movies:
            half_number_of_movies = number_of_movies // 2
            unknown_liked_movies = {a[0] for a in top_movies[:half_number_of_movies]}
            known_liked_movies = top_movies[half_number_of_movies:]
            recs, label = s.kmeans.get_reccomendations_for_user(known_liked_movies, half_number_of_movies)
            recs = s.cf_user.get_reccomendations_for_user(label, known_liked_movies, [])
            recommendations_set = set(recs)
            jaccard_index = len(unknown_liked_movies.intersection(recommendations_set)) / len(unknown_liked_movies.union(recommendations_set))
            jaccard_indexes.append(jaccard_index)
            if jaccard_indexes:
                t_users.set_description(f"Jaccard eval: {sum(jaccard_indexes) / len(jaccard_indexes)}")
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Indeks Jaccarda', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.savefig('accuracy_2.png')
    plt.clf()


def accuracy_3(s):
    d = s.dataset
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 10000)
    jaccard_indexes = []
    t_users = tqdm(test_users)
    for user in t_users:
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        number_of_movies = 20
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:number_of_movies]
        if len(top_movies) == number_of_movies:
            half_number_of_movies = number_of_movies // 2
            unknown_liked_movies = {a[0] for a in top_movies[:half_number_of_movies]}
            known_liked_movies = top_movies[half_number_of_movies:]
            recs, label = s.kmeans.get_reccomendations_for_user(known_liked_movies, half_number_of_movies)
            recs = s.cf_user.get_reccomendations_for_user(label, known_liked_movies, [])
            recommendations_set = set(recs)
            jaccard_index = len(unknown_liked_movies.intersection(recommendations_set)) / len(unknown_liked_movies.union(recommendations_set))
            jaccard_indexes.append(jaccard_index)
            if jaccard_indexes:
                t_users.set_description(f"Jaccard eval: {sum(jaccard_indexes) / len(jaccard_indexes)}")
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Indeks Jaccarda', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.savefig('accuracy_3.png')
    plt.clf()


def accuracy_all(s):
    d = s.dataset
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 10000)
    jaccard_indexes = []
    t_users = tqdm(test_users)
    for user in t_users:
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        number_of_movies = 30
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:number_of_movies]
        if len(top_movies) == number_of_movies:
            half_number_of_movies = number_of_movies // 2
            third_number_of_movies = number_of_movies // 3
            unknown_liked_movies = {a[0] for a in top_movies[:half_number_of_movies]}
            known_liked_movies = top_movies[half_number_of_movies:]
            rec1, rec2, rec3 = s.get_raw_recommendations_for_user(known_liked_movies)
            recs = rec1[:third_number_of_movies] + rec2[:third_number_of_movies] + rec3[:third_number_of_movies]
            recommendations_set = set(recs)
            jaccard_index = len(unknown_liked_movies.intersection(recommendations_set)) / len(unknown_liked_movies.union(recommendations_set))
            jaccard_indexes.append(jaccard_index)
            if jaccard_indexes:
                t_users.set_description(f"Jaccard eval: {sum(jaccard_indexes) / len(jaccard_indexes)}")
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Indeks Jaccarda', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.savefig('accuracy_all.png')
    plt.clf()


def variety_1(s):
    d = Dataset()
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 10000)
    jaccard_indexes = []
    all_vectors = [movie.genres_vector for movie in d.all_movies]
    global_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*all_vectors)]
    global_average = [i / sum(global_average) for i in global_average]
    number_of_movies = 10
    for user in tqdm(test_users, desc="EVALUATING ACCURACY"):
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:number_of_movies]
        if len(top_movies) == number_of_movies:
            recs, label = s.kmeans.get_reccomendations_for_user(top_movies, number_of_movies)
            recs = s.cf_user.get_reccomendations_for_user(label, top_movies, [])
            recommendations_set = set(recs)
            recommendations = [movie.genres_vector for movie in d.all_movies if movie.movieId in recommendations_set]
            column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*recommendations)]
            variety_index = sum([(column_average[i] - global_average[i]) ** 2 for i in range(len(column_average))])
            jaccard_indexes.append(variety_index)
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Różnorodność', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.savefig('variety_1.png')
    plt.clf()


def variety_2(s):
    d = Dataset()
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 10000)
    jaccard_indexes = []
    all_vectors = [movie.genres_vector for movie in d.all_movies]
    global_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*all_vectors)]
    global_average = [i / sum(global_average) for i in global_average]
    number_of_movies = 10
    for user in tqdm(test_users, desc="EVALUATING ACCURACY"):
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:number_of_movies]
        if len(top_movies) == number_of_movies:
            recs, _ = s.kmeans.get_reccomendations_for_user(top_movies, number_of_movies)
            recommendations_set = set(recs)
            recommendations = [movie.genres_vector for movie in d.all_movies if movie.movieId in recommendations_set]
            column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*recommendations)]
            variety_index = sum([(column_average[i] - global_average[i]) ** 2 for i in range(len(column_average))])
            jaccard_indexes.append(variety_index)
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Różnorodność', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.savefig('variety_2.png')
    plt.clf()


def variety_3(s):
    d = Dataset()
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 10000)
    jaccard_indexes = []
    all_vectors = [movie.genres_vector for movie in d.all_movies]
    global_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*all_vectors)]
    global_average = [i / sum(global_average) for i in global_average]
    number_of_movies = 10
    for user in tqdm(test_users, desc="EVALUATING ACCURACY"):
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:number_of_movies]
        if len(top_movies) == number_of_movies:
            recs, _ = s.kmeans.get_reccomendations_for_user(top_movies, number_of_movies)
            recommendations_set = set(recs)
            recommendations = [movie.genres_vector for movie in d.all_movies if movie.movieId in recommendations_set]
            column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*recommendations)]
            variety_index = sum([(column_average[i] - global_average[i]) ** 2 for i in range(len(column_average))])
            jaccard_indexes.append(variety_index)
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Różnorodność', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.savefig('variety_3.png')
    plt.clf()


def variety_all(s):
    d = Dataset()
    all_users = d.get_all_user_ids()
    test_users = random.sample(all_users, len(all_users) // 10000)
    jaccard_indexes = []
    all_vectors = [movie.genres_vector for movie in d.all_movies]
    global_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*all_vectors)]
    global_average = [i / sum(global_average) for i in global_average]
    number_of_movies = 10
    for user in tqdm(test_users, desc="EVALUATING ACCURACY"):
        rated_movies = [a for a in d.ratings_by_user[user] if a[1] > 0.5]
        top_movies = sorted(rated_movies, key=lambda x: x[1], reverse=True)[:number_of_movies]
        if len(top_movies) == number_of_movies:
            third_number_of_movies = number_of_movies // 3
            rec1, rec2, rec3 = s.get_raw_recommendations_for_user(top_movies)
            recs = rec1[:third_number_of_movies] + rec2[:third_number_of_movies] + rec3[:third_number_of_movies]
            recommendations_set = set(recs)
            recommendations = [movie.genres_vector for movie in d.all_movies if movie.movieId in recommendations_set]
            column_average = [sum(sub_list) / len(sub_list) for sub_list in zip(*recommendations)]
            variety_index = sum([(column_average[i] - global_average[i]) ** 2 for i in range(len(column_average))])
            jaccard_indexes.append(variety_index)
    sns.set_style('whitegrid')
    k = sns.kdeplot(jaccard_indexes)
    k.set(xlabel='Różnorodność', ylabel='Gęstość')
    print(sum(jaccard_indexes) / len(jaccard_indexes))
    plt.savefig('variety_all.png')
    plt.clf()


syst = System()
variety_1(syst)
variety_2(syst)
variety_3(syst)
variety_all(syst)
