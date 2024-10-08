###########################################
# Item-Based Collaborative Filtering
###########################################

# Veri seti: https://grouplens.org/datasets/movielens/

# Adım 1: Veri Setinin Hazırlanması
# Adım 2: User Movie Df'inin Oluşturulması
# Adım 3: Item-Based Film Önerilerinin Yapılması
# Adım 4: Çalışma Scriptinin Hazırlanması

######################################
# Adım 1: Veri Setinin Hazırlanması
######################################
import pandas as pd
pd.set_option('display.max_columns', 500)

movie = pd.read_csv('Miuul/5. Hafta (Tavsiye Sistemleri)/recommender_systems/datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('Miuul/5. Hafta (Tavsiye Sistemleri)/recommender_systems/datasets/movie_lens_dataset/rating.csv')
df = movie.merge(rating, how="left", on="movieId") # movie sol tarafta rating sağ tarafta
df.head()

######################################
# Adım 2: User Movie Df'inin Oluşturulması
######################################

df.shape
df["title"].nunique()
df["title"].value_counts().head() # filmlerin kaç tane rating olacağını bulacağız.

comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["count"] <= 1000].index

common_movies = df[~df["title"].isin(rare_movies)]
# rare movies'in içindeki title'ları seç, onun dışındaki al

common_movies.shape
common_movies["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
# satırlara user ları, sütunlara title'ları, values kısmına rating leri aldık

user_movie_df.shape

######################################
# Adım 3: Item-Based Film Önerilerinin Yapılması
######################################

movie_name = "Matrix, The (1999)"
movie_name = "Ocean's Twelve (2004)"
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
# title'lar arasından 1 tane örnek alıyoruz ve gelen değerin str kısmını
# istediğimizden dolayı 0. index
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]

check_film("Sherlock", user_movie_df)


######################################
# Adım 4: Çalışma Scriptinin Hazırlanması
######################################

def create_user_movie_df():
    import pandas as pd
    movie = pd.read_csv('Miuul/5. Hafta (Tavsiye Sistemleri)/recommender_systems/datasets/movie_lens_dataset/movie.csv')
    rating = pd.read_csv('Miuul/5. Hafta (Tavsiye Sistemleri)/recommender_systems/datasets/movie_lens_dataset/rating.csv')
    df = movie.merge(rating, how="left", on="movieId")
    comment_counts = pd.DataFrame(df["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = df[~df["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")
    return user_movie_df

user_movie_df = create_user_movie_df()


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending=False).head(10)

item_based_recommender("Matrix, The (1999)", user_movie_df)

movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

item_based_recommender(movie_name, user_movie_df)





