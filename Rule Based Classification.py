
#############################################
# Kural Tabanlı Sınıflandırma ile Potansiyel Müşteri Getirisi Hesaplama
#############################################

#############################################
# İş Problemi
#############################################
# Bir oyun şirketi müşterilerinin bazı özelliklerini kullanarak seviye tabanlı (level based) yeni müşteri tanımları (persona)
# oluşturmak ve bu yeni müşteri tanımlarına göre segmentler oluşturup bu segmentlere göre yeni gelebilecek müşterilerin şirkete
# ortalama ne kadar kazandırabileceğini tahmin etmek istemektedir.

# Örneğin: Türkiye’den IOS kullanıcısı olan 25 yaşındaki bir erkek kullanıcının ortalama ne kadar kazandırabileceği belirlenmek isteniyor.


#############################################
# Veri Seti Hikayesi
#############################################
# Persona.csv veri seti uluslararası bir oyun şirketinin sattığı ürünlerin fiyatlarını ve bu ürünleri satın alan kullanıcıların bazı
# demografik bilgilerini barındırmaktadır. Veri seti her satış işleminde oluşan kayıtlardan meydana gelmektedir. Bunun anlamı tablo
# tekilleştirilmemiştir. Diğer bir ifade ile belirli demografik özelliklere sahip bir kullanıcı birden fazla alışveriş yapmış olabilir.

# Price: Müşterinin harcama tutarı
# Source: Müşterinin bağlandığı cihaz türü
# Sex: Müşterinin cinsiyeti
# Country: Müşterinin ülkesi
# Age: Müşterinin yaşı

################# Uygulama Öncesi #####################

#    PRICE   SOURCE   SEX COUNTRY  AGE
# 0     39  android  male     bra   17
# 1     39  android  male     bra   17
# 2     49  android  male     bra   17
# 3     29  android  male     tur   17
# 4     49  android  male     tur   17

################# Uygulama Sonrası #####################

#       customers_level_based        PRICE SEGMENT
# 0   BRA_ANDROID_FEMALE_0_18  1139.800000       A
# 1  BRA_ANDROID_FEMALE_19_23  1070.600000       A
# 2  BRA_ANDROID_FEMALE_24_30   508.142857       A
# 3  BRA_ANDROID_FEMALE_31_40   233.166667       C
# 4  BRA_ANDROID_FEMALE_41_66   236.666667       C

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)


#############################################
# PROJE GÖREVLERİ
#############################################

#############################################
# GÖREV 1: Aşağıdaki soruları yanıtlayınız.
#############################################

# Soru 1: persona.csv dosyasını okutunuz ve veri seti ile ilgili genel bilgileri gösteriniz.
df = pd.read_csv("persona.csv")
df.head()
df.info()

df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
# Soru 2: Kaç unique SOURCE vardır? Frekansları nedir?
df["SOURCE"].nunique()
df["SOURCE"].value_counts()
# Soru 3: Kaç unique PRICE vardır?
df["PRICE"].nunique()

# Soru 4: Hangi PRICE'dan kaçar tane satış gerçekleşmiş?
df["PRICE"].value_counts()

# Soru 5: Hangi ülkeden kaçar tane satış olmuş?
df["COUNTRY"].value_counts()
df.groupby(["COUNTRY"])["PRICE"].count()
df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="count")


# Soru 6: Ülkelere göre satışlardan toplam ne kadar kazanılmış?
df.groupby(["COUNTRY"])["PRICE"].sum()
df.groupby(["COUNTRY"]).agg({"PRICE": "sum"})
df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="sum")


# Soru 7: SOURCE türlerine göre göre satış sayıları nedir?
df.groupby(["SOURCE"])["PRICE"].count()
df["SOURCE"].value_counts()
df.pivot_table(values="PRICE",index="SOURCE",aggfunc="count")


# Soru 8: Ülkelere göre PRICE ortalamaları nedir?
df.groupby(["COUNTRY"])["PRICE"].mean()
df.groupby(["COUNTRY"]).agg({"PRICE": "mean"})
df.pivot_table(values="PRICE",index="COUNTRY",aggfunc="mean")


# Soru 9: SOURCE'lara göre PRICE ortalamaları nedir?
df.groupby(["SOURCE"])["PRICE"].mean()
df.groupby(["SOURCE"]).agg({"PRICE": "mean"})
df.pivot_table(values="PRICE",index="SOURCE",aggfunc="mean")


# Soru 10: COUNTRY-SOURCE kırılımında PRICE ortalamaları nedir?
df.groupby(["SOURCE", "COUNTRY"])["PRICE"].mean()
df.groupby(["SOURCE", "COUNTRY"]).agg({"PRICE": "mean"})
df.pivot_table(values="PRICE",index=["SOURCE","COUNTRY"],aggfunc="mean")


#############################################
# GÖREV 2: COUNTRY, SOURCE, SEX, AGE kırılımında ortalama kazançlar nedir?
#############################################
df.groupby(["SOURCE", "COUNTRY", "SEX", "AGE"])["PRICE"].mean()
df.groupby(["SOURCE", "COUNTRY", "SEX", "AGE"]).agg({"PRICE": "mean"})
df.pivot_table(values="PRICE",index=["SOURCE","COUNTRY", "SEX", "AGE"],aggfunc="mean")

#############################################
# GÖREV 3: Çıktıyı PRICE'a göre sıralayınız.
#############################################
# Önceki sorudaki çıktıyı daha iyi görebilmek için sort_values metodunu azalan olacak şekilde PRICE'a uygulayınız.
# Çıktıyı agg_df olarak kaydediniz.
agg_df = df.groupby(["SOURCE", "COUNTRY", "SEX", "AGE"])["PRICE"].mean().sort_values(ascending=False)

agg_df = df.groupby(["COUNTRY", 'SOURCE', "SEX", "AGE"]).agg({"PRICE": "mean"}).sort_values("PRICE", ascending=False)


#############################################
# GÖREV 4: Indekste yer alan isimleri değişken ismine çeviriniz.
#############################################
# Üçüncü sorunun çıktısında yer alan PRICE dışındaki tüm değişkenler index isimleridir.
# Bu isimleri değişken isimlerine çeviriniz.
# İpucu: reset_index()
# agg_df.reset_index(inplace=True)

agg_df = agg_df.reset_index()
agg_df.head()


#############################################
# GÖREV 5: AGE değişkenini kategorik değişkene çeviriniz ve agg_df'e ekleyiniz.
#############################################
# Age sayısal değişkenini kategorik değişkene çeviriniz.
# Aralıkları ikna edici olacağını düşündüğünüz şekilde oluşturunuz.
# Örneğin: '0_18', '19_23', '24_30', '31_40', '41_70'
#df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])

df["AGE"].nunique()

mylabels = ['0_18', '19_23', '24_30', '31_40', '41_' + str(agg_df["AGE"].max())]

agg_df["AGE_CAT"] = pd.cut(agg_df["AGE"], [0, 18, 23, 30, 40, agg_df["AGE"].max()], labels=mylabels)
agg_df.head()


#############################################
# GÖREV 6: Yeni level based müşterileri tanımlayınız ve veri setine değişken olarak ekleyiniz.
#############################################
# customers_level_based adında bir değişken tanımlayınız ve veri setine bu değişkeni ekleyiniz.
# Dikkat!
# list comp ile customers_level_based değerleri oluşturulduktan sonra bu değerlerin tekilleştirilmesi gerekmektedir.
# Örneğin birden fazla şu ifadeden olabilir: USA_ANDROID_MALE_0_18
# Bunları groupby'a alıp price ortalamalarını almak gerekmektedir.

agg_df['customers_level_based'] = agg_df[['COUNTRY', 'SOURCE', 'SEX', 'AGE_CAT']].agg(lambda x: '_'.join(x).upper(), axis=1)
agg_df.head()
agg_df["customers_level_based"].nunique()

# Birçok aynı segment olacak.
# kontrol edelim:
agg_df["customers_level_based"].value_counts()
agg_df[agg_df["customers_level_based"] == "BRA_ANDROID_MALE_24_30"]

# Bu sebeple segmentlere göre groupby yaptıktan sonra price ortalamalarını almalı ve segmentleri tekilleştirmeliyiz.
agg_df = agg_df.groupby("customers_level_based").agg({"PRICE": "mean"})
agg_df.head(10)  # şu anda customers_level_based indexte, bunu değişkene çevirmek gerekir.

agg_df = agg_df.reset_index()
agg_df.head()

# kontrol edelim. her bir persona'nın 1 tane olmasını bekleriz:
agg_df["customers_level_based"].value_counts()
agg_df.head()



#############################################
# GÖREV 7: Yeni müşterileri (USA_ANDROID_MALE_0_18) segmentlere ayırınız.
#############################################
# PRICE'a göre segmentlere ayırınız,
# segmentleri "SEGMENT" isimlendirmesi ile agg_df'e ekleyiniz,
# segmentleri betimleyiniz,

agg_df["SEGMENT"] = pd.qcut(agg_df["PRICE"], 4, labels=["D", "C", "B", "A"])
agg_df.head(30)
agg_df.groupby("SEGMENT").agg({"PRICE": ["mean","min", "max", "sum", "count"]})
agg_df.head()


# utku:
agg_df["SEGMENT_"] = pd.cut(agg_df["PRICE"], bins=3, labels=["düşük", "orta", "yüksek"])
segment_betimi = agg_df.groupby('SEGMENT_').agg({'PRICE': ['mean', 'max', 'min', 'count']}).reset_index()




#############################################
# GÖREV 8: Yeni gelen müşterileri sınıflandırınız ne kadar gelir getirebileceğini tahmin ediniz.
#############################################
# 33 yaşında ANDROID kullanan bir Türk kadını hangi segmente aittir ve ortalama ne kadar gelir kazandırması beklenir?

new_user = "TUR_ANDROID_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user]
agg_df.shape
agg_df["customers_level_based"].unique()



# 35 yaşında IOS kullanan bir Fransız kadını hangi segmente ve ortalama ne kadar gelir kazandırması beklenir?
new_user2 = "FRA_IOS_FEMALE_31_40"
agg_df[agg_df["customers_level_based"] == new_user2]
agg_df.shape
