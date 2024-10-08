import numpy as np
# 1. Kategorik değişken: sütun grafik. Countplot bar
# 2. Sayısal değişken: histogram, boxplot

#KATEGORİK DEĞİŞKEN GÖRSELLEŞTİRME

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
df = sns.load_dataset("titanic")
df.head()


df["sex"].value_counts().plot(kind='bar')
plt.show()

# SAYISAL DEĞİŞKEN GÖRSELLEŞTİRME
plt.hist(df["age"])
plt.show()


plt.boxplot(df["fare"])
plt.show()


# MATPLOTLİB ÖZELLİKLERİ

#PLOT

x = np.array([1, 8])
y = np.array([0, 150])

plt.plot(x, y)
plt.show()

plt.plot(x, y, 'o')
plt.show()


x = np.array([2,4,6,8,10])
y = np.array([1,3,5,7,9])
plt.plot(x, y)
plt.show()
plt.plot(x, y, 'o')
plt.show()



# MARKER

y = np.array([13,28,11,100])

plt.plot(y, marker='o')
plt.show()


# LİNE

y = np.array([13,28,11,100])

plt.plot(y, ls = "dashed", color="r")
plt.show()

# MULTİPLE LİNES

x = np.array([13,18,31,10])
y = np.array([13,28,11,100])
plt.plot(x)
plt.plot(y)
plt.show()

# LABELS

x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([240,250,260,270,280,290,300,310,320,330])
plt.plot(x, y)

plt.title("Bu ana başlık")
plt.xlabel("x ekseni")
plt.ylabel("y ekseni")
plt.grid()
plt.show()


x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([240,250,260,270,280,290,300,310,320,330])
plt.subplot(1, 2, 1)
plt.title("1")
plt.plot(x, y)


x = np.array([80,85,90,95,100,105,110,115,120,125])
y = np.array([240,250,260,270,280,290,300,310,320,330])
plt.subplot(1, 2, 2)
plt.title("2")
plt.plot(x, y)


# SEABORN İLE GÖRSELLEŞTİRME

df = sns.load_dataset("tips")
df.head()
df["sex"].value_counts()
sns.countplot(x=df["sex"], data=df)
plt.show()


sns.boxplot(x=df["total_bill"])
plt.show()

df["total_bill"].hist()
plt.show()