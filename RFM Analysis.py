# RFM Analysis
# RFM: Recency, Frequency, Monetary
# müşteri segmentasyonu için kullanılan bir teknik
#satın alma alışkanlıklarına göre gruplara ayrılma

#Recency: Yenilik müşterinin en son yaptığı alışveriş(av)

#Frequency: Sıklık

# Monetary: Parasal değer

#RFM skoru r,f,m değerlerinin yan yana gelerek string olarak geldiği skorlar
# 145 örnek

#SKORLAR ÜZERİNDEN SEGMENTLER OLUŞTURMAK
# Bir e ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre
# pazarlama stratejilerini belirlemek istiyor.

# Değişkenler
#
# InvoiceNo: Fatura numarası. Her işleme yani faturaya ait eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode: Ürün kodu. Her bir ürün için eşsiz numara.
# Description: Ürün ismi
# Quantity: Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# InvoiceDate: Fatura tarihi ve zamanı.
# UnitPrice: Ürün fiyatı (Sterlin cinsinden)
# CustomerID: Eşsiz müşteri numarası
# Country: Ülke ismi. Müşterinin yaşadığı ülke.

# Veriyi anlama
import datetime as dt
import pandas as pd
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df_ = pd.read_excel("Miuul/3. Hafta/online_retail_II.xlsx", sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape
df.isnull().sum()
df["Description"].nunique() # eşşiz ürün sayısı

df["Description"].value_counts().head()
df.groupby(["Description"]).agg({"Quantity": "sum"}).head()

df.groupby(["Description"]).agg({"Quantity": "sum"}).sort_values("Quantity", ascending=False).head()

df["Invoice"].nunique()

df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()

df.groupby(["Invoice"]).agg({"TotalPrice": "sum"}).head()


# VERİ HAZIRLAMA

# customer id'lerde çok fazla null var. dropluyoruz.
df.isnull().sum()
df.dropna(inplace=True)  # eksik değerleri uçuruyoruz.
df.shape

df.describe().T
# negatif değerler var. iadelerden kaynaklanıyor. veri setinden çıkarıcaz

df = df[~df["Invoice"].str.contains("C", na=False)]

# 4. RFM Metriklerinin Hesaplanması(recency, frequency, monetary)

df.head()

df["InvoiceDate"].max()
today_date = dt.datetime(2010, 12, 11)
type(today_date)

rfm = df.groupby(["Customer ID"]).agg({"InvoiceDate": lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                       "Invoice": lambda Invoice: Invoice.nunique(),
                                       "TotalPrice": lambda TotalPrice: TotalPrice.sum()})

# Invoice date: Recency için
# Invoice: Frequency için. farklı fatura sayısı müşterinin ne kadar tane av yaptığını gösterir
# TotalPrice: Monetary için

rfm.columns = ["Recency", "Frequency", "Monetary"]
rfm.head()
rfm.describe().T

rfm = rfm[rfm["Monetary"] > 0] # Monetary değerinin 0 olması iyi bişey değilmiş???
rfm.head()
rfm.describe().T

# 5. RFM Skorlarının Hesaplanması

rfm["recency_score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])

rfm["monetary_score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])

rfm["frequency_score"] = pd.qcut(rfm["Frequency"].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
# rank methodu frequencyde çok fazla aynı değer olduğu için kullanıldı.
rfm.head()

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))
# recency ve frequency skorlarını astype ile stringe çevirip birleştirdik.
rfm.head()

rfm.describe().T

# 6. RFM Segmentlerinin Oluşturulması ve Analizi

# Regex
# RFM isimlendirilmesi
seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}

rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
rfm.head()

rfm[["segment", "Recency", "Frequency", "Monetary"]].groupby("segment").agg(["mean", "count"])

rfm[rfm["segment"] == "need_attention"].head()
rfm[rfm["segment"] == "need_attention"].index

new_df = pd.DataFrame()

new_df["new_customer_id"] = rfm[rfm["segment"] == "new_customers"].index

new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)
# sonundaki 0'ları yok ettik int'e çevirerek

new_df.to_csv("new_customer_id")

# 7. Tüm Sürecin Fonksiyonlaştırılması


def create_rfm(dataframe, csv=False):

    #VERİYİ HAZIRLAMA
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na=False)]

    # RFM METRIKLERININ HESAPLANMASI
    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('Customer ID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                'Invoice': lambda num: num.nunique(),
                                                "TotalPrice": lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', "monetary"]
    rfm = rfm[(rfm['monetary'] > 0)]

    # RFM SKORLARININ HESAPLANMASI
    rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # cltv_df skorları kategorik değere dönüştürülüp df'e eklendi
    rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                        rfm['frequency_score'].astype(str))

    # SEGMENTLERIN ISIMLENDIRILMESI
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)
    rfm = rfm[["recency", "frequency", "monetary", "segment"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm


df = df_.copy()

rfm_new = create_rfm(df, csv=True)


