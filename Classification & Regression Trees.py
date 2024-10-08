

import matplotlib
matplotlib.use('TkAgg')
import warnings
import joblib
import pydotplus
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt

from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error

from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate, cross_val_score, validation_curve
from sklearn.model_selection import cross_val_score, RandomizedSearchCV

from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


# dataframe in sınırlamalarını kaldırıyoruz

pd.set_option('display.max_columns', None)  # bütün sütunları göster
pd.set_option('display.max_rows', None)     # bütün satırları göster
# pd.reset_option('display.max_rows')
pd.set_option('display.width', 500)  # sütunlar max 500 tane gösterilsin
pd.set_option('display.expand_frame_repr', False)  # çıktının tek bir satırda olmasını sağlar

pd.set_option('display.float_format', lambda x: '%.3f' % x)  # virgülden sonra 3 basamak gösterir



"""""
GÖREV 1 : Veri setine EDA işlemlerini uygulayınız.

Genel Resim
Kategorik Değişken Analizi (Analysis of Categorical Variables)
Sayısal Değişken Analizi (Analysis of Numerical Variables)
Hedef Değişken Analizi (Analysis of Target Variable)
Korelasyon Analizi (Analysis of Correlation)
Adım 1: Train ve Test veri setlerini okutup birleştiriniz. Birleştirdiğiniz veri üzerinden ilerleyiniz.
"""""

# train ve test setlerinin bir araya getirilmesi.
train = pd.read_csv("Miuul/7. Hafta (Makine Öğrenmesi)/CART exer/train (1).csv")
test = pd.read_csv("Miuul/7. Hafta (Makine Öğrenmesi)/CART exer/test (1).csv")
train["SalePrice"]
test.columns

df = pd.concat([train, test], ignore_index=False).reset_index()
# ignore_index=False parametresi, orijinal DataFrame'lerin indekslerini korumak için kullanılır.
# Eğer True olsaydı, yeni birleştirilmiş DataFrame için yeni indeksler oluşturulacaktı
# df = pd.concat([train, test], ignore_index=True).reset_index(drop=True) şeklinde kullanarak da index'siz bir df yaratılabilir

df.head()
df.tail()
df.shape


df = df.drop("index", axis=1)

# axis=1 parametresi, silinecek şeyin bir sütun olduğunu belirtir (axis=0 satırlar için kullanılır)
# Bu satır, df DataFrame'inden "index" adlı sütunu siler (drop fonksiyonu ile).

# Kısaca bu adım, bir önceki adımda eklenen eski indeks sütununu kaldırır, çünkü genellikle analizde gerekli değildir.

df.head()


df.isnull().sum()


# Dataframe in genel resmine tek seferde bakıyoruz

def check_df(dataframe, exclude_object = True):                  # exclude_object=True object tipindeki sütunları hariç tutar.
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(3))
    print("##################### Tail #####################")
    print(dataframe.tail(3))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())

    # Sadece numeric kolonlar için quantile hesapla
    if exclude_object:
        dataframe = dataframe.select_dtypes(exclude=['object'])        # select_dtypes belirli veri tiplerine sahip sütunları seçmek için kullanılır
                                                                       #select_dtypes(include=[...]) dersen yalnızca içermesini istediğin veri tiplerini gir.
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


check_df(df)


# bu fonksiyon bize gizli numerikleri ve gizli kategorikleri gösterecek
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    grab_col_names for given dataframe

    :param dataframe:
    :param cat_th:
    :param car_th:
    :return:
    """

    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]

    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]

    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    print(f'cat_but_car: {len(cat_but_car)}')

    # cat_cols + num_cols + cat_but_car = değişken sayısı.
    # num_but_cat cat_cols'un içerisinde zaten.
    # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
    # num_but_cat sadece raporlama için verilmiştir.

    return cat_cols, num_cols, num_but_cat, cat_but_car

# df veri setimizi grab_col_names fonksiyonundan geçiriyoruz
cat_cols, num_cols, num_but_cat, cat_but_car = grab_col_names(df)

df['BedroomAbvGr'].head()

print(f"Kategorik Kolonlar: cat_cols: 52")
print("################################")
print(cat_cols)
print("-------------------------------------------------------------------")
print("                                                                   ")
print(f"Numerik Kolonlar: num_cols: 28")
print("################################")
print(num_cols)
print("-------------------------------------------------------------------")
print("                                                                   ")
print(f"Numerik Gözüken Kategorik Kolonlar: num_but_cat: 10")
print("################################")
print(num_but_cat)
print("-------------------------------------------------------------------")
print("                                                                   ")
print(f"Kategorik Gözüken Kardinal Kolonlar: cat_but_car: 1")
print("################################")
print(cat_but_car)
print("-------------------------------------------------------------------")
print("                                                                   ")




# KATEGORİK DEĞİŞKENLERİN ANALİZİ
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)





# NUMERİK DEĞİŞKENLERİN ANALİZİ
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col, False)



"""
# KATEGORİK DEĞİŞKENLERİN TARGET'A GÖRE ANALİZİ
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col).agg({'target': ["mean", "median"]})}), end="\n\n\n")
"""

def target_summary_with_cat(dataframe, target, categorical_col):
    # Veri çerçevesini gruplama ve özet istatistikler oluşturma
    summary_df = dataframe.groupby(categorical_col).agg({target: ['count', 'mean', 'median' ]})

    summary_df.columns = ['target_count','target_mean', 'target_median']

    summary_df['target_mean'] = summary_df['target_mean'].apply(lambda x: int(round(x)) if pd.notna(x) else x)
    summary_df['target_median'] = summary_df['target_median'].apply(lambda x: int(round(x)) if pd.notna(x) else x)

    print(summary_df, end="\n\n\n")



for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)




for col in cat_cols:
    target_summary_with_cat(df,"SalePrice",col)



# Bağımlı değişkenin incelenmesi histogram grafiği
# bağımlı değişkenin normal dağılmasını bekleriz ancak bir sağa çarpıklık söz konusu solda yığılma var
# bu dağılımı normalleştirmek için bazı işlemler yapabilmekteyiz
# df["SalePrice"].hist(bins=100)
plt.hist(df["SalePrice"], bins=100, edgecolor='black')
#plt.ylabel('frequency')
plt.xlabel('Sale Price')
plt.show()




# Varyans katsayısını hesaplama

coeff_variance = {}
for column in df.columns:
    if column != "Id":
        if pd.api.types.is_numeric_dtype(df[column]): #Check if the column is numeric
            mean = df[column].mean()
            std = df[column].std()
            coeff_variance[column] = (std / mean) * 100
        else:
            coeff_variance[column] = "N/A" # Handle non-numeric columns

coeff_variance

sorted_coeff_variance = dict(sorted(coeff_variance.items(), key=lambda item: (item[1] if isinstance(item[1], (int, float)) else float('-inf')), reverse=True))

# Sonuçları yazdırma
print(sorted_coeff_variance)




"""""
“Coefficient of Variation”

Değişim Katsayısı olarak adlandırılan Bulanıklık Katsayısı diye de isimlendirilen CoV, veri setindeki değişkenliğin (varyansın) ölçüsüdür.

Değişim katsayısını hesaplamak için şu formül kullanılır:

Değişim Katsayısı (CV) = (Standart Sapma / Ortalama) x 100

Standart Sapma, veri setinin dağılımındaki yayılmanın ölçüsüdür.

Ortalama ise veri setindeki değerlerin aritmetik ortalamasıdır.

Değişim katsayısı ne kadar küçükse, veri setinin o kadar homojen olduğu ve değişkenliğin az olduğu söylenebilir. Aksine, yüksek bir değişim katsayısı, veri setinde daha fazla değişkenlik olduğunu gösterir.

Burada yorum yapabilmek için alınması gereken eşik değer %35 değeridir.

(CV) = (Standart Sapma / Ortalama) x 100 < %35 ise tablo nettir, veri seti normal dağılımlıdır yada normal dağılıma yakındır yorumunu yaparız. Eşik değerden aşağı inildikçe veri seti homojen olmaya daha da yaklaşır.

(CV) = (Standart Sapma / Ortalama) x 100 > %35 ise de veri dağınıktır, bulanıktır içinde farklı segmentleri barındırır ve ihtimaller dahilinde aykırı değerler bulundurabilir yorumunu yaparız.

"""""



# Bağımlı değişkenin logaritmasının incelenmesi
# logartimik dönüşümü bağımlı değişkene uyguluyoruz, bu dönüşüm dağılımı biraz daha normalleştiriyor.
# model kurarken bağımlı değişkenin logaritmik dönüştürülmüş haliyle işlem yapabiliriz
np.log1p(df['SalePrice']).hist(bins=50)
plt.show()




# 5.Korelasyon Analizi (Analysis of Correlation)


# değişkenler arasındaki ilişkiyi incelemek için korelasyon analizine bakalım
# korelasyon iki değişken arasındaki ilişkinin yönünü ve derecesini gösterir
# -1 ile +1 arasında değişir ve 0 herhangi bir ilişki olmadığını gösterir
# -1 e yaklaştıkça negatif güçlü ilişki, +1 e yaklaştıkça pozitif güçlü ilişki olduğunu gösterir

corr = df[num_cols].corr()
corr["LotArea"].sort_values(ascending=False)



# Korelasyonların gösterilmesi
# renk kırmızıya doğru kaydıkça negatif güçlü ilişki artmaktadır,
# renk koyu maviye doğru kaydıkça da pozitif güçlü ilişki artmaktadır
sns.set(rc={'figure.figsize': (15, 15)})
sns.heatmap(corr, cmap="RdBu")
plt.show()



# KORELASYON ANALİZİ FARKLI BİR GÖSTERİM (tipi numerik olanlar ile)
df[num_cols].corr()

# Korelasyon Matrisi
f, ax = plt.subplots(figsize=[18, 13])
sns.heatmap(df[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="Blues")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show()




# bağımlı değişken ile bağımsız değişken arasında güçlü ilişki olsun isteriz. Çünkü bağımsız değişken bağımlı değişkeni etkileyen onun hakkında bilgi veren ve onu açıklayan değişkenlerdir.
# Ancak bağımsız değişkenler arasında çok fazla güçlü ilişki olmasını istemeyiz çünkü birbirinden etkilenmesini istemeyiz
# etkilenmesi durumu da bize çoklu doğrusal bağlantı sorununa yol açar. Bunu istemeyiz ancak bu regresyon modeli için geçerlidir. diğer durumlarda bu bağlantı gözardı edilebilmektedir.



def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    # Select only numeric columns for correlation calculation
    numeric_df = dataframe.select_dtypes(include=['number'])

    # Calculate the correlation matrix
    corr = numeric_df.corr()
    # Take the absolute value of the correlation matrix
    cor_matrix = corr.abs()
    # Create an upper triangle matrix to avoid duplicate checks
    # (Korelasyon matrisinin üst üçgeni oluşturulur, böylece aynı korelasyon çiftlerinin tekrar tekrar kontrol edilmesi önlenir.)
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    # Find columns with correlation above the threshold [Korelasyonu verilen eşikten (corr_th) yüksek olan sütunlar bulunur.]
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    # Plot the heatmap if requested
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr,annot=True, fmt=".2f", annot_kws={"size": 8},  cmap="RdBu")
        #ax.set_title("Correlation Matrix", fontsize=20)
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True)


# yüksek korelasyona sahip değişkenler
# ['1stFlrSF', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageArea', 'SalePrice']



# Birbirleriyle yüksek korelasyona sahip bu değişkenleri drop_list adında bir listeye kaydediyoruz
# NOT: Bu liste, korelasyonu belirlenen eşiğin (0.70) üzerinde olan sütunları içerir.
# Bu sütunlar modelde çoklu doğrusal bağlantı sorununa neden olabileceği için düşürülmesi önerilir
drop_list = high_correlated_cols(df, plot=False)
print(drop_list)



# Bu işlem, SalePrice ile diğer tüm sütunlar arasındaki korelasyon katsayılarının mutlak değerlerini alıp,
# bu değerleri azalan sırayla sıralayarak en güçlü ilişkileri en üste getirir.
saleprice_corr = corr['SalePrice'].abs().sort_values(ascending=False)
print(saleprice_corr)

# En yüksek korelasyona sahip sütun SalePrice ile kendisidir (1.000000), ardından OverallQual (0.790982), GrLivArea (0.708624) ve diğerleri gelir.



top_corr_features = saleprice_corr.index[:10]
print("En yüksek korelasyona sahip sütunlar: ", top_corr_features)

# Bu çıktı, SalePrice ile en yüksek korelasyona sahip ilk 10 sütunu listeler.
# Bu sütunlar, modelde önemli açıklayıcı değişkenler olarak kullanılabilir, çünkü hedef değişken (SalePrice) ile güçlü ilişkileri vardır.
# Ancak, bu değişkenlerin birbirleriyle olan korelasyonlarına da dikkat edilmelidir,
# çünkü bu yüksek korelasyonlar çoklu doğrusal bağlantı (multicollinearity) sorunlarına yol açabilir
top_corr_matrix = df[top_corr_features].corr()
print(top_corr_matrix)
sns.heatmap(top_corr_matrix,annot=True, cmap="RdBu")
plt.show()

#Görev 2 : Feature Engineering

#Aykırı Değer Analizi

# Aykırı değerlerin baskılanması
def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit


# baskılama aykırı değerlerin en alt değerlere ve en üst değerlere göre sabitlenmesi durumudur

# thresholdumuzu veri setimize göre gözlem sayısına göre, değişkenlerin yapısına göre kendi know-how ımıza göre belirleyebiliriz
# genel geçer %75 e %25 şeklinde alınandır. ancak çok fazla bilgi kaybetmemek için bu değerleri büyütmek mümkündür
# fazlaca baskılama yapmak çoğu zaman bilgi kaybına ve değişkenlerin arasındaki ilişkinin kaybolmasına neden olabilmektedir

# Aykırı değer kontrolü

# bu eşik değerlerlere göre aykırı değerler var mı değişkenlerde, varsa hangilerinde var kontrol edeceğiz
# bir değişkenin aykırı değerlerini bool olarak sorgulatacağız
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


for col in num_cols:
    if col != "SalePrice":
      print(col, check_outlier(df, col))



# Aykırı değerlerin baskılanması
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df,col)


# tekrar bakalım aykırı değer kalmış mı
for col in num_cols:
    print(col, check_outlier(df, col))



#Eksik Değer Analizi

df_subset = df.iloc[:, :10]

plt.figure(figsize=(14, 8))
msno.bar(df_subset)
plt.show()
# "Alley", "PoolQC", "Fence", "MiscFeature"  çok fazla eksik gözlem var
# sokak erişim türü, havuz kalitesi, çit kalitesi, Diğer kategorilerde yer almayan çeşitli özellikler




# bu fonksiyonla elimize bir veri geldiğinde bu veriyi hızlı bir şekilde eksikliklerin frekansı nedir
# hangi değişkenlerde eksiklik var ve bu eksikliklerin oranı nedir bilgisini göreceğiz

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])       # [n_miss, np.round(ratio, 2)]   hangi değerlerin olacağını söyler
                                                                                                 # axis=1  birleştirmenin sütunlar boyunca olacaını söyler
                                                                                                 # keys=['n_miss', 'ratio']    bunlar sütun isimleri
    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)



#
df["Alley"].value_counts()
#
df["BsmtQual"].value_counts()


# Bazı değişkenlerdeki boş değerler evin o özelliğe sahip olmadığını ifade etmektedir,
# bu kanıya data seti iyice inceleyerek ve data setin ve değişkenlerin dinamiklerine bakarak karar vermeliyiz
# örneğin PoolQC bir gözlemde boş ise o evde havuz olmadığını belirtmektedir.
no_cols = ["Alley","BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","PoolQC","Fence","MiscFeature"]





# Kolonlardaki boşlukların "No" ifadesi ile doldurulması
# burada değişkenler kendi nezdinde incelemeli hepsine medyan ya da mod ya da ortalama uygulamak yerine
# değişken bazında uygun metrik ile doldurmak daha uygun olacaktır
for col in no_cols:
    df[col].fillna("No",inplace=True)

missing_values_table(df)



"""""
#Eksik veri problemi nasıl çözülür ###########################################

Silme : eksik verilerin değişkenlerinin silinmesi

dropna diyerek silebiliriz ama bu durumda gözlem sayısı azalacaktır. gözlem sayısı çok fazla ve eksik gözlem sayısı az ise eksik değerler silinebilir ancak gözlem sayısının az olduğu ya da eksik değerlerin fazlaca bulunduğu verilerde silme işlemi yapmak ciddi oranda bir veri kaybına yol açacaktır

Değer atama yöntemleri : ortalama mod medyan gibi basit atama yöntemleri

değişkenlerdeki eksiklikleri medyanı ya da ortalamasıyla doldurabileceğimiz gibi her hangi bir sabit değerle de doldurabiliriz

Tahmine dayalı yöntemler : ML ya da istatistiksel bazı yöntemlerle tahminlere göre değer atama ################################################################################

biz şimdi mode medyan yöntemleri ile atamaya geçiş yapacağız

nümerik değişkenlerin eksiklerinin tamamlanması

Bu fonsksiyonun ön tanımlı değeri medyandır. bunu daha sonra num_method="XXX" girerek değiştirebiliriz

eksik değerlerin median veya mean ile doldurulmasını sağlar

categorik değişkenler için eşik 20 belirlenmiştir bu değişkenin sahip olabileceği maksimum

eşsiz sınıf sayısını ifade eder. Varsayılan target değer de "SalePrice" şeklindedir.

Fonksiyon önce veri kümesindeki eksik değerlere sahip değişkenleri tanımlar ve bunları variables_with_na adlı bir listede saklar. Daha sonra hedef değişkeni temp_target adlı geçici bir değişkende saklar.

Ardından fonksiyon, veri kümesindeki her sütuna bir lambda işlevi uygulamak için Apply() yöntemini kullanır.

Lambda işlevi, her sütunun veri tipini ve benzersiz değerlerinin sayısını kontrol eder ve eksik değerleri şu şekilde doldurur:

Veri türü "O" (yani nesne) ise ve benzersiz değerlerin sayısı cat_length=20 küçük veya ona eşitse, mod (yani en sık kullanılan değer) ile eksik değerler atanır.

Eğer num_method "mean" ise, "O" dışında veri tipine sahip sütunlardaki eksik değerler ortalama değerle ilişkilendirilir.

Eğer num_method "medyan" ise, "O" dışında veri tipine sahip sütunlardaki eksik değerler medyan değerle hesaplanır.

Son olarak, işlev hedef değişkeni geri yükler ve atamadan önce ve sonra her sütundaki eksik değerlerin sayısını yazdırır.

Daha sonra değiştirilen veri kümesini döndürür.

"""""




# Bu fonsksiyon eksik değerlerin median veya mean ile doldurulmasını sağlar

def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data


df = quick_missing_imp(df, num_method="median", cat_length=17)


"""""
# BEFORE
MSZoning           4
LotFrontage      486
Utilities          2
Exterior1st        1
Exterior2nd        1
MasVnrType      1766
MasVnrArea        23
BsmtFinSF1         1
BsmtFinSF2         1
BsmtUnfSF          1
TotalBsmtSF        1
Electrical         1
BsmtFullBath       2
BsmtHalfBath       2
KitchenQual        1
Functional         2
GarageYrBlt      159
GarageCars         1
GarageArea         1
SaleType           1
SalePrice       1459
dtype: int64 


# AFTER 
 Imputation method is 'MODE' for categorical variables!
 Imputation method is 'MEDIAN' for numeric variables! 

MSZoning           0
LotFrontage        0
Utilities          0
Exterior1st        0
Exterior2nd        0
MasVnrType         0
MasVnrArea         0
BsmtFinSF1         0
BsmtFinSF2         0
BsmtUnfSF          0
TotalBsmtSF        0
Electrical         0
BsmtFullBath       0
BsmtHalfBath       0
KitchenQual        0
Functional         0
GarageYrBlt        0
GarageCars         0
GarageArea         0
SaleType           0
SalePrice       1459
dtype: int64 

"""""


# tekrar kontrol edelim
missing_values_table(df)
# ve hiç eksiklik kalmadı, SalePrice hariç



# eksik değerlerin olduğu değişkenlere bakalım
[col for col in df.columns if df[col].isnull().sum() > 0]
# Out[23]: ['SalePrice'] sale hariç kalmadı




# Rare analizi yapınız ve rare encoder uygulayınız.


# buradaki kategorik değişkenleri seçmemiz lazım grab col names i çağıracağız
# kategorik değişkenleri getiriyoruz
# neden bunu yapıyoruz
# gereksiz sayıda kategori olmasın ve benzer kategorileri bir araya getirelim ya da işe yaramayanları çıkartalım
cat_cols, num_cols, num_but_cat, cat_but_car = grab_col_names(df)




# şimdi bu kategorik değişkenlerimizi ve sınıflarını, sınıfların azlık çokluk durumlarına göre analiz edelim

# 1. Kategorik değişkenlerin azlık çokluk durumunun analiz edilmesi.
######################################################################
# şimdi burda yani bir fonksiyona ihtiyacımız var ve bu fonksiyonla öyle bir işlem yapmamız lazım ki
# kategorik değişkenlerin sınıflarını ve bu sınıfların oranlarını getirsin, plot=true dersek grafikler de gelir

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

# bunu neden yaptık

# rare az gözlemlenen nadir demektir
# az gözlemlenen değişkenler one hot encoding işlemi yaptığımızda sütununda fazla bilgi barındırmayacaktır
# bu nedenle modellemede karşılık bulamayacaklardır
# bu nedenle one hot encoderdan geçirip değişken haline getirdiğimiz değerlerin de ölçüm kalitesinin olmasını
# ve bağımlı değişkene çeşitli olası etkilerini göz önünde bulundurmak isteriz.
# bu nedenle gereksiz değişkenlerden uzaklaşmak kurtulmak için rare encoder ı kullanabiliriz
# toparlamak gerekirse veri setindeki bir kategorik değişkenin sınıflarındaki az değerlerden kurtulmak için
# bir eşik değeri belirleriz ve bu belirlediğimiz belirli bir eşik değerine göre altta kalan sınıfları
# toparlarız bir araya getirip bunlara rare deriz yani bir bakıma dışarda bırakırız




# Rare kategoriler ile bağımlı değişken arasındaki ilişkinin analiz edilmesi.

# şimdi bu rare kategorisine alacağımız sınıfların değişkenlerin bağımlı değişkene etkileri nedir ve
# arasındaki ilişki nasıldır bunu analiz edeceğiz

# bunu neden yapıyoruz? gereksiz sayıda kategori olmasın, birbirlerine benzeyen yerleri, değişkenleri veya kategorileri olabildiğince bir araya getirelim
# bilgi vermeyen sınıfları bir araya getirelim bilgi vermeyen sınıflardan kurtulalım ya da onları başka bir kategoriye dahil edelim

# bu fonskiyonla sınıfların frekansları oranları ve target yani SalePrice açıdından ortalamaları gelecek

# Kategorik kolonların dağılımının incelenmesi
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        # Bu satır, her kategorik sütunun kaç farklı kategoriye sahip olduğunu yazdırır.
        print(col, ":", len(dataframe[col].value_counts()))

        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
        # COUNT: Her kategorinin kaç kez tekrarlandığını (frekans) gösterir.
        # RATIO: Her kategorinin toplam veri kümesindeki oranını hesaplar.
        # TARGET_MEAN: Hedef değişkenin (SalePrice) her kategori için ortalama değerini hesaplar.
rare_analyser(df, "SalePrice", cat_cols)




# Analiz

# MSZoning : 5:

# MSZoning değişkeninin 5 farklı kategoriye sahip olduğunu gösterir: C (all), FV, RH, RL, RM.

# COUNT:
# C (all): 25 kez tekrarlanmış.
# FV: 139 kez tekrarlanmış.
# RH: 26 kez tekrarlanmış.
# RL: 2269 kez tekrarlanmış.
# RM: 460 kez tekrarlanmış.

# RATIO:
# C (all): Veri kümesinin %0.9'unu oluşturur.
# FV: Veri kümesinin %4.8'ini oluşturur.
# RH: Veri kümesinin %0.9'unu oluşturur.
# RL: Veri kümesinin %77.7'sini oluşturur.
# RM: Veri kümesinin %15.8'ini oluşturur.

# TARGET_MEAN:
# C (all): Ortalama SalePrice 74,528.
# FV: Ortalama SalePrice 214,014.062.
# RH: Ortalama SalePrice 131,558.375.
# RL: Ortalama SalePrice 191,004.995.
# RM: Ortalama SalePrice 126,316.830.

# Sonuç
# MSZoning değişkeninde C (all) ve RH kategorileri nadir kategorilerdir (her biri veri kümesinin sadece %0.9'unu oluşturur).
# RL kategorisi veri kümesinin büyük bir kısmını (%77.7) oluşturur ve SalePrice için yüksek bir ortalamaya sahiptir.
# FV kategorisi, nadir olmayan ama yine de SalePrice için en yüksek ortalamaya sahip kategoridir.
# RM kategorisi, nispeten daha yaygın olup, ortalama SalePrice daha düşüktür.
# Bu analiz, nadir kategorileri belirlemeye ve gerekirse bu nadir kategorileri birleştirerek veya
# başka kategorilere dahil ederek daha anlamlı ve istatistiksel olarak güçlü bir veri kümesi oluşturmanıza yardımcı olabilir.
# Bu şekilde, veri kümesinde fazla kategori bulunması ve bilgi vermeyen sınıfların etkisi azaltılmış olur.



# Rare encoder'ın yazılması.

# rare yüzdemizi belirleyeceğiz bu oranın altında kalan kategorik değişken sınıflarını bir araya getirecek.
# rare encoderımız veri setindeki seyrek sınıflı kategorik değişkenlerin seyrek sınıflarını toplayıp
# bir araya getirerek bunlara rare isimlendirmesi yapmaktadır

# Nadir sınıfların tespit edilmesi
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    # Orijinal veri kümesini değiştirmemek için bir kopyasını oluşturduk.

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    # Kategorik değişkenler (tür olarak O - object) arasında nadir kategorilere sahip olanları belirler.
    # Bir kategorinin nadir sayılması için frekans oranının rare_perc'den küçük olması gerekmektedir.

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
        # Her nadir kategorik sütun için:
        # Kategorilerin frekans oranlarını hesaplar.
        # Nadir kategorileri belirler (frekans oranı rare_perc'den küçük olanlar).
        # Bu nadir kategorileri "Rare" etiketiyle değiştirir.

    return temp_df.head(100)




rare_encoder(df, 0.01)

# rare_encoder(df, 0.01) fonksiyonunu çağırdığınızda, df veri kümesindeki nadir kategoriler 0.01 (yani %1) frekans oranının altında olanlar olarak belirlenir ve
# bu kategoriler "Rare" etiketiyle değiştirilir.

rare_analyser(df, "SalePrice", cat_cols)



# Before
"""MSZoning : 5
          COUNT  RATIO  TARGET_MEAN
MSZoning
C (all)      25  0.009    74528.000
FV          139  0.048   214014.062
RH           26  0.009   131558.375
RL         2269  0.777   191004.995
RM          460  0.158   126316.830"""

# MSZoning değişkeninde C (all) ve RH kategorileri 0.01'den daha az bir oran (%0.9) ile nadir olarak sayılacak ve "Rare" etiketi ile değiştirilecektir.

# After
"""MSZoning : 4
          COUNT  RATIO  TARGET_MEAN
MSZoning
FV          139  0.048   214014.062
RL         2269  0.777   191004.995
RM          460  0.158   126316.830
Rare         51  0.018   (Bu kategoriler için yeni bir ortalama hesaplanacaktır)"""

# Bu işlem, veri kümesinde nadir kategorilerin sayısını azaltarak modelin daha kararlı ve anlamlı sonuçlar üretmesini sağlar.
# Ayrıca, nadir kategorilerden kaynaklanabilecek gürültüyü azaltarak modelin genelleme yeteneğini artırır.

df["MSZoning"].dtype



# dff diye kaydediyoruz bütün kategorik değişkenlerin sınıflarını rare encoderdan geçirdikten sonra
# böylelikle df te bir sorun olduğunda her şeyi baştan çalıştırmak yeine burdan yeni oluşturduğumuz dff ile devam edebiliriz

dff = rare_encoder(df, 0.01)



# Rare altında topladıktan sonra tekrar bakalım rare analiz çıktımıza bir kontrol edelim rare kategorileri
rare_analyser(dff, "SalePrice", cat_cols)
dff.head()

# kontrollerden sonra yine df ile devam ediyoruz. Her ihtimale karşı elimizde tutmak için buraya kadar yapılan işlemlerden oluşan dff yedek dataframe elimizde var.




# FEATURE ENGINEERING

# Yeni değişkenler oluşturunuz ve oluşturduğunuz yeni değişkenlerin başına 'NEW' ekleyiniz.



df["NEW_1st*GrLiv"] = df["1stFlrSF"] * df["GrLivArea"]

df["NEW_Garage*GrLiv"] = (df["GarageArea"] * df["GrLivArea"])

# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2

# Porch Area
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF

df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF


# Lot Ratio
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea

df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea

# MasVnrArea
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea

# Dif Area
df["NEW_DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF)


df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]


df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt

df["NEW_HouseAge"] = df.YrSold - df.YearBuilt

df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd

df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt

df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd)

df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt




df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1)



# kolonlar üzerinden yeni feature lar ürettik ve eskilerine gerek kalmadı bu yüzden bunlara ihtiyacımız yok ve data frame den düşüreceğiz
drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

# drop_list'teki değişkenlerin düşürülmesi
df.drop(drop_list, axis=1, inplace=True)



# kolonlar üzerinden yeni feature lar ürettik ve eskilerine gerek kalmadı bu yüzden bunlara ihtiyacımız yok ve data frame den düşüreceğiz
drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope","Heating", "PoolQC", "MiscFeature","Neighborhood"]

# drop_list'teki değişkenlerin düşürülmesi
df.drop(drop_list, axis=1, inplace=True)


df.shape




# Label Encoding & One-Hot Encoding işlemlerini uygulayınız.

# Değişkenlerin tiplerine göre ayrılması işlemi yeni değişkenlerden sonra
cat_cols, num_cols, num_but_cat, cat_but_car = grab_col_names(df)



# label encoding / binary encoding işlemini 2 sınıflı kategorik değişkenlere uyguluyoruz
# yani nominal sınıflı kategorik değişkenlere böylelikle bu iki sınıfı 1-0 şeklinde encodelamış oluyoruz

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)


df.head()


# one-hot encoder ise ordinal sınıflı kategorik değişkenler için uyguluyoruz. sınıfları arasında fark olan
# değişkenleri sınıf sayısınca numaralandırıp kategorik değişken olarak df e gönderiyor

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)



df.head()
df.shape


# MODELLEME

# GÖREV 3: Model kurma


#  Train ve Test verisini ayırınız. (SalePrice değişkeni boş olan değerler test verisidir.)
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]
# Train verisi ile model kurup, model başarısını değerlendiriniz.
# bağımlı ve bağımsız değişkenleri seçiyoruz
train_df.head()
test_df.head()


# Sale price çarpık bir dağılıma sahipti, öncelikle log dönüşümü yapmadan modelleme kuracağız
# daha sonra da log dönüşümü yaparak model kuracağız ve rmse değerlerimizi log öncesi ve log sonrasına göre karşılaştıracağız
y = train_df['SalePrice'] # np.log1p(df['SalePrice'])  y= bağımlı değişken
X = train_df.drop(["Id", "SalePrice"], axis=1)        # X = Id hariç bağımsız değişkenler (90 değişkenle beraber)
# Train verisi ile model kurup, model başarısını değerlendiriniz.
# modelimizi kuruyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
# kullanacağımız yöntemi import ettik
from lightgbm import LGBMRegressor

# bağımlı değişkenimiz sayısal ise regression, regressor algoritmalarını         LGBMRegressor
# bağımlı değişkenimiz kategorikse classification algoritmalarını kullanıyoruz   LGBMClassifier

# kullanacağımız yöntemleri içeren bir model tanımlı nesne kuruyoruz
# kapalı olan algoritmaları da açarak onları da modele sokabilirsiniz
models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]



# daha sonra model nesnemizi döngü ile rmse değerini her bir yöntem için hesaplayacak şekilde
# fonksiyonel olarak çağırıyoruz

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

# RMSE: 42501.7985 (LR)       RMSE: 42577.6248 (LR)      sağdakiler benim çıktılar
                           #  RMSE: 33094.0411(Ridge)
                           #  RMSE: 41907.6612 (Lasso)
                           #  RMSE: 33983.4208 (ElasticNet)
# RMSE: 47557.3947 (KNN)      RMSE: 47557.3947 (KNN)
# RMSE: 38786.4984 (CART)     RMSE: 40467.3613 (CART)
# RMSE: 28910.3004 (RF)       RMSE: 29441.8337 (RF)
                           #  RMSE: 81072.8172 (SVR)
# RMSE: 25725.1132 (GBM)      RMSE: 25719.7779 (GBM)
# RMSE: 27971.7767 (XGBoost)  RMSE: 28023.3716 (XGBoost)
# RMSE: 28582.004 (LightGBM)  RMSE: 28489.3618 (LightGBM)
# RMSE: 25551.3003 (CatBoost) RMSE: 25185.6671 (CatBoost)


df['SalePrice'].mean()        # 180921
df['SalePrice'].std()         # 79442
1/(df['SalePrice'].mean()/df['SalePrice'].std())  # 0.439   0.35'ten büyük. veri dağınık

"""""
Standart Sapma (Standard Deviation) Standart sapma, veri noktalarının ortalamadan ne kadar farklılık gösterdiğinin bir ölçüsüdür. Yani, veri setindeki değerlerin dağılımının ne kadar yaygın olduğunu gösterir. Standart sapma ne kadar büyükse, veri noktaları ortalamadan o kadar çok sapar ve dağılım o kadar geniştir.

Ortalama ve Standart Sapma Arasındaki Fark ve Kıyaslama Ortalama, veri setinin merkezi eğilimini temsil ederken, standart sapma veri noktalarının bu merkeze olan ortalama uzaklığını temsil eder. Birlikte, veri setinin genel yapısını ve dağılımını anlamamıza yardımcı olurlar.

Düşük Standart Sapma: Eğer standart sapma değeri düşükse, bu veri noktalarının ortalamaya yakın olduğunu ve veri setinin oldukça homojen olduğunu gösterir. Yani, veri noktaları birbirine benzer ve tutarlıdır.

Yüksek Standart Sapma: Yüksek standart sapma, veri noktalarının ortalamadan büyük ölçüde sapmalar gösterdiğini ve veri setinin heterojen olduğunu gösterir. Veri noktaları arasında büyük farklılıklar olabilir ve veri seti daha değişkendir.

Kıyaslama Yöntemi: Ortalama ve standart sapmayı kıyaslamak için doğrudan bir "fark" hesaplaması genellikle yapılmaz. Bunun yerine, standart sapmanın büyüklüğünü ortalamaya göre değerlendiririz. Örneğin, ortalamaya oranla standart sapmanın büyük veya küçük olması, veri dağılımının yaygınlığı hakkında bilgi verir.

BONUS : Log dönüşümü yaparak model kurunuz ve rmse sonuçlarını gözlemleyiniz.

Not: Log'un tersini (inverse) almayı unutmayınız.
"""""


# Log dönüşümünün gerçekleştirilmesi

# tekrardan Train ve Test verisini ayırıyoruz.
train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]
# Bağımlı değişkeni normal dağılıma yaklaştırarak model kuracağız

y = np.log1p(train_df['SalePrice'])
X = train_df.drop(["Id", "SalePrice"], axis=1)
# Verinin eğitim ve test verisi olarak bölünmesi
# log dönüşümlü hali ile model kuruyoruz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)
# lgbm_tuned = LGBMRegressor(**lgbm_gs_best.best_params_).fit(X_train, y_train)
# bağımlı değişkendeki log dönüştürülmüş tahminlemelere bakıyoruz

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
y_pred
# Bağımlı değişkendeki gözlemlerin tahminlemiş halleri geliyor (log dönüştürülmüş halleri geldi tabi)
# gerçek değerlerle karşılaştırma yapabilmek için bu log dönüşümünün tekrar tersini (inverse) almamız gerekmektedir.

# daha sonra model nesnemizi döngü ile rmse değerini her bir yöntem için hesaplayacak şekilde
# fonksiyonel olarak çağırıyoruz

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")


# # LOG DÖNÜŞÜMÜ ÖNCESİ
# RMSE: 42501.7985 (LR)
# RMSE: 47557.3947 (KNN)
# RMSE: 38786.4984 (CART)
# RMSE: 28910.3004 (RF)
# RMSE: 25725.1132 (GBM)
# RMSE: 27971.7767 (XGBoost)
# RMSE: 28582.004 (LightGBM)
# RMSE: 25551.3003 (CatBoost)

# LOG DÖNÜŞÜMÜ SONRASI
# RMSE: 0.1547 (LR)            # RMSE: 0.1545 (LR)
                               # RMSE: 0.1328 (Ridge)
                               # RMSE: 0.1792 (Lasso)
                               # RMSE: 0.1586 (ElasticNet)
# RMSE: 0.2357 (KNN)             RMSE: 0.2357 (KNN)
# RMSE: 0.2047 (CART)            RMSE: 0.2085 (CART)
# RMSE: 0.1419 (RF)              RMSE: 0.1411 (RF)
                               # RMSE: 0.2236 (SVR)
# RMSE: 0.1301 (GBM)             RMSE: 0.1287 (GBM)
# RMSE: 0.1427 (XGBoost)         RMSE: 0.1437 (XGBoost)
# RMSE: 0.1343 (LightGBM)        RMSE: 0.1353 (LightGBM)
# RMSE: 0.1239 (CatBoost)        RMSE: 0.1233 (CatBoost)



# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması (y_pred için)
new_y = np.expm1(y_pred)
new_y
np.expm1(0.1545)
# burada y_pred değerleri log dönüşümü yapılmış hedef değişken tahminlerini gösterirken
# new_y değeri y_pred in inverse uygulanmış yani log dönüşümünün tersinin yapılmış halinin tahmin sonuçlarını göstermektedir.
# bu iki değerlerin çıktılarını yani log dönüşümlü ve dönüşümsüz hallerini karşılaştırabilirsiniz



# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması (y_test için)
new_y_test = np.expm1(y_test)
new_y_test


# Inverse alınan yani log dönüşümü yapılan tahminlerin RMSE değeri
np.sqrt(mean_squared_error(new_y_test, new_y))    # 24259.4444


"""""
Log dönüşümü ve ardından yapılan inverse log dönüşümü (log dönüşümünün tersi), veri biliminde özellikle regresyon modellerinde sıkça karşılaşılan bir tekniktir. Bu tekniklerin kullanılmasının başlıca nedenleri şunlardır:

Veri Dağılımını Düzeltmek: Gerçek dünyada karşılaşılan birçok veri seti, normal dağılımdan sapmalar gösterir. Özellikle, hedef değişkenin sağa ya da sola çarpık olduğu durumlar, lineer regresyon gibi bazı algoritmaların varsayımlarını ihlal edebilir. Log dönüşümü, çarpık veriyi daha normal bir dağılıma dönüştürerek bu algoritmaların daha iyi performans göstermesine yardımcı olabilir.

Hata Terimlerinin Varyansını Sabitlemek: Regresyon modelleri için bir diğer önemli varsayım, hata terimlerinin sabit bir varyansa (homoscedasticity) sahip olmasıdır. Çarpık verilerde, büyük değerlere sahip gözlemler genellikle daha büyük hata terimlerine sahip olabilir. Log dönüşümü, bu varyansı sabitleyerek modelin daha tutarlı tahminler yapmasına olanak tanır.

Çok Büyük veya Çok Küçük Değerlerle Başa Çıkmak: Bazı durumlarda, hedef değişkende çok büyük veya çok küçük değerler olabilir. Bu tür değerler, modelin öğrenme sürecini olumsuz etkileyebilir. Log dönüşümü, değer aralığını sıkıştırarak bu sorunu hafifletebilir.

Veriye log dönüşümü uygulandıktan sonra, modelin tahminleri de log dönüşümlü hedef değişken üzerinde yapılmış olur. Ancak, gerçek dünya uygulamalarında tahminlerin orijinal ölçeğe dönüştürülmesi gerekir. Bu nedenle, modelin çıkışındaki tahminlerin log dönüşümünün tersi alınarak orijinal ölçeğe dönüştürülmesi gerekir. Bu işlem, np.expm1 fonksiyonu ile gerçekleştirilir. np.expm1(x) fonksiyonu, exp(x) - 1 hesaplamasını yapar ve bu, log1p dönüşümünün (yani log(1+x)) tersidir.

Modelin performansını değerlendirirken, tahminlerin ve gerçek değerlerin orijinal ölçeğe dönüştürülmesi, elde edilen hata metriğinin (örneğin, RMSE) daha anlamlı ve yorumlanabilir olmasını sağlar. Çünkü son kullanıcılar veya karar vericiler, modelin çıktılarını ve performansını orijinal ölçekte anlamak isteyecektir.
"""""


# Hiperparametre optimizasyonlarını gerçekleştiriniz.

lgbm_model = LGBMRegressor(random_state=46)
rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# rmse: 0.13433133803712316            0.13528215696923862
# bu henüz hiç bir hiperparametre ayarlaması yapılmamış base modelin rmse sonucudur,
# aşağıda hiperparametre optimizasyonu yaptıktan sonra tekrar bir rmse değeri bakacağız ve bu değerle onu karşılaştır. Düşüş gözlemlenmeli

lgbm_model.get_params()


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}


lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)
# lgbm_gs_best.best_params_ =  {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 500}



rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# rmse: 0.1296004503932677            0.13048666629472735      eskisi= 0.13528215696923862
# bu hiperparametre optimizasyonu yapılmış final modelin rmse değeridir



# ŞİMDİ BİR DE CATBOOST İLE MODEL KURALIM


catboost_model = CatBoostRegressor(random_state=17)
rmse = np.mean(np.sqrt(-cross_val_score(catboost_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# rmse: 0.12285753813282337             0.12382815322942844
# catboost base modelinin rmse sonucu



catboost_model.get_params()

# {'loss_function': 'RMSE', 'random_state': 17}


"""""
CatBoostRegressor ve LGBMRegressor gibi farklı makine öğrenmesi kütüphanelerinin get_params() metodunun farklı çıktılar vermesi, bu kütüphanelerin implementasyonları ve varsayılan parametrelerin nasıl yönetildiğiyle ilgilidir. Her iki kütüphane de Python'da sınıflar aracılığıyla implemente edilmiştir ve get_params() metodu, bir sınıfın anlık (instance) özelliklerini (yani parametrelerini) bir sözlük olarak döndürür. Ancak, bir kütüphanenin get_params() metodu çağrıldığında hangi parametrelerin görüntüleneceği, o kütüphanenin nasıl tasarlandığına bağlıdır.

CatBoost CatBoostRegressor'ın get_params() metodunun sadece {'loss_function': 'RMSE', 'random_state': 46} gibi sınırlı bir çıktı vermesinin nedeni, CatBoost'un yalnızca değiştirilmiş veya açıkça belirtilmiş parametreleri döndürmesidir. Yani, eğer bir parametre varsayılan değerini koruyorsa ve bu değer CatBoost tarafından içsel olarak yönetiliyorsa, bu parametre get_params() çıktısında görünmeyebilir. CatBoost, kullanıcının belirtmediği parametreler için genellikle içsel varsayılan değerleri kullanır ve bu yüzden get_params() çıktısı daha minimal olabilir.

LightGBM Öte yandan, LGBMRegressor'un get_params() metodunun daha fazla parametre bilgisi vermesi, LightGBM'in varsayılan parametrelerini açık bir şekilde kullanıcının erişimine sunmasıyla ilgilidir. LightGBM, oluşturulduğu anda tüm varsayılan parametreleri açıkça belirler ve bunları get_params() çıktısında döndürür.

Hiperparametre Ayarlama Bu farklılık, hiperparametre ayarlama kabiliyetinizi etkilemez. Her iki kütüphane de, modelinizi oluştururken veya modelinizi oluşturduktan sonra hiperparametreleri ayarlamanıza olanak tanır. CatBoostRegressor ve LGBMRegressor için hiperparametre optimizasyonu yapabilir ve model performansınızı iyileştirebilirsiniz. Örneğin, Grid Search, Random Search veya Bayesian Optimization gibi yöntemlerle en iyi parametre setini bulabilirsiniz.

CatBoost için hiperparametre ayarlama yaparken, dökümantasyonda belirtilen tüm parametreleri inceleyebilir ve ihtiyaçlarınıza göre bunları ayarlayabilirsiniz. Ayarladığınız parametreler, get_params() metodu çağrıldığında döndürülen sözlükte görünecektir.

Sonuç olarak, get_params() metodunun farklı çıktıları, kütüphanelerin tasarım farklılıklarından kaynaklanmaktadır ve hiperparametre ayarlama yeteneğinizi etkilemez. Her iki kütüphane de geniş bir hiperparametre setini destekler ve bu parametreler üzerinde optimizasyon yaparak model performansını artırabilirsiniz.
"""""


# CatBoost'un resmi dokümantasyonu, kullanılabilir tüm parametreleri ve her birinin varsayılan değerlerini içerir.
#CatBoostRegressor ve diğer CatBoost modelleri için parametre referanslarına dokümantasyondan ulaşabilirsiniz.

# Python'un yerleşik help() fonksiyonunu kullanarak CatBoostRegressor sınıfının dokümantasyonuna erişebilir ve parametreleri hakkında bilgi alabilirsiniz.
# Bu yöntem, interaktif Python oturumları veya Jupyter Notebook'larda hızlı bir şekilde parametreleri gözden geçirmek için kullanışlıdır.
from catboost import CatBoostRegressor
help(CatBoostRegressor)



# Yukarıdan çektiğimiz bazı parametrelere göre yeni bir catboost hiperparametre ayarlamaları yapalım
catboost_params = {
    "n_estimators": [100, 500],
    "learning_rate": [0.01, 0.1],
    "max_depth": [2, 3],
    "random_state": [17],  # Modelin tekrarlanabilirliğini sağlamak için
    # Daha fazla parametre eklenebilir, örneğin:
    "l2_leaf_reg": [1, 3, 5],  # Düzenlileştirme terimi
    # "border_count": [32, 64, 128],  # Sayısal özellikler için sınır sayısı (bölme noktaları)
    # "auto_class_weights": ['Balanced', None],  # Sınıflar arası dengesizliği düzeltmek için
    # "bootstrap_type": ['Bayesian', 'Bernoulli', 'MVS'],  # Örnekleme yöntemi
    # Özelleştirilebilir başka parametreler...
}



catboost_gs_best = GridSearchCV(catboost_model,
                            catboost_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

catboost_gs_best.best_params_

# 'l2_leaf_reg': 5,  'learning_rate': 0.1, 'max_depth': 3,  'n_estimators': 500, 'random_state': 17

# Değişkenlerin önem düzeyini belirten feature_importance fonksiyonunu kullanarak özelliklerin sıralamasını çizdiriniz.

# feature importance
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:50])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")
#lightgbm modeli ile plot importance grafiğini çıkartıyoruz
model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X)



#catboost modeli ile plot importance grafiğini çıkartıyoruz
model2 = CatBoostRegressor()
model2.fit(X, y)

plot_importance(model2, X)



# buradan catboost çıktılarıyla lightgbm çıktılarının görselleri karşılaştırılarak hangi modelin daha uygun olduğuna ve değişkenlere yanıt verdiğine dair bir yorumlama yapabiliriz


# test dataframe indeki boş olan salePrice değişkenlerini tahminleyiniz
#
# Kaggle sayfasına submit etmeye uygun halde bir dataframe oluşturunuz. (Id, SalePrice)



# şimdi test veri setinde hedef değişkenimizin tahminlemesini yapıyoruz
model = LGBMRegressor()
model.fit(X, y)

# test veri setinden Id ve SalePrice sütunlarını çıkarıp ve geriye kalan sütunları modelin tahmin yapması için kullanacağız
# eğitilmiş modeli kullanarak test veri seti üzerinde tahmin yapıyoruz. Test veri setinden "Id" ve "SalePrice" sütunları çıkarıyoruz çünkü bu tahminleri yapmak gereksizdir.
# "Id" sütunu, her veri noktasının benzersiz kimliğidir ve "SalePrice" hedef değişkendir, tahmin edilmeye çalışılan değerdir.
predictions = model.predict(test_df.drop(["Id","SalePrice"], axis=1))




# şimdi test veri setinde hedef değişkenimizin tahminlemesini yapıyoruz
model = LGBMRegressor()
model.fit(X, y)

# test veri setinden Id ve SalePrice sütunlarını çıkarıp ve geriye kalan sütunları modelin tahmin yapması için kullanacağız
# eğitilmiş modeli kullanarak test veri seti üzerinde tahmin yapıyoruz. Test veri setinden "Id" ve "SalePrice" sütunları çıkarıyoruz çünkü bu tahminleri yapmak gereksizdir.
# "Id" sütunu, her veri noktasının benzersiz kimliğidir ve "SalePrice" hedef değişkendir, tahmin edilmeye çalışılan değerdir.
predictions = model.predict(test_df.drop(["Id","SalePrice"], axis=1))
predictions1 = np.expm1(predictions)



dictionary = {"Id":test_df.index, "SalePrice":predictions1}   # bir sözlük oluşturduk. Bu sözlük, test veri setinin indekslerini "Id" olarak ve tahmin edilen değerleri "SalePrice" olarak içeriyor. Bu, her tahminin hangi evle ilişkili olduğunu belirlemek için kullanılır.
dfSubmission = pd.DataFrame(dictionary)  #  sözlüğü bir pandas DataFrame'ine dönüştürüyoruz
dfSubmission.to_csv("housePricePredictions1.csv", index=False)  #  tahmin sonuçlarını "housePricePredictions.csv" adlı bir CSV dosyasına kaydediyoruz. index=False parametresi, DataFrame'in indekslerinin dosyaya kaydedilmemesini sağlar.
dff1 = pd.read_csv("housePricePredictions1.csv")
dff1["SalePrice"].mean()
dff1['Id'] = dff1['Id'].astype(int)
dff1.to_csv("housePricePredictions3.csv", index=False)

dff1.at[1458, 'Id'] = 2918
dff1.to_csv("housePricePredictions4.csv", index=False)
dff1.index = dff1.index + 1
dff1["Id"] = dff1["Id"] +1
dff1.to_csv("housePricePredictions5.csv", index=False)

mean_sale_price = dff1['SalePrice'].mean()
new_id = dff1['Id'].max() + 1
new_row = pd.DataFrame({'Id': [new_id], 'SalePrice': [mean_sale_price]})
dff2 = pd.concat([dff1, new_row], ignore_index=True)

dff2['Id'] = dff2['Id'].astype(int)

dff2.to_csv("housePricePredictions2.csv", index=False)
df



rf_model = RandomForestRegressor(random_state=46)
rmse = np.mean(np.sqrt(-cross_val_score(rf_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# rmse:0.14093704013706837
# bu henüz hiç bir hiperparametre ayarlaması yapılmamış base modelin rmse sonucudur,
# aşağıda hiperparametre optimizasyonu yaptıktan sonra tekrar bir rmse değeri bakacağız ve bu değerle onu karşılaştır. Düşüş gözlemlenmeli

rf_model.get_params()


rf_params = {"max_depth": [5, 8, None],
             "max_features": [3, 5, 7, "sqrt"],
             "min_samples_split": [2, 5, 8, 15, 20],
             "n_estimators": [100, 200, 500]}


rf_best_grid = GridSearchCV(rf_model,
                            rf_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)

rf_best_grid.best_params_
# 'max_depth': None,  'max_features': 'sqrt', 'min_samples_split': 2,  'n_estimators': 200

rf_final_model = rf_model.set_params(**rf_best_grid.best_params_, random_state=46).fit(X, y)


rmse = np.mean(np.sqrt(-cross_val_score(rf_final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
rmse
# 0.13953376935629538          eskisi=0.14093704013706837