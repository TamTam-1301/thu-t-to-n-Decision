import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# đọc dữ liệu
df = pd.read_csv(r"C:\Users\Admin\Documents\Visual Studio 2022\Thuật toán Decision tree\data.csv"  )
df.head()

df["classìication"].value_counts
df['Classification'] = df['Classification'] - 1
df['Classification'].value_counts()
 #Tạo dữ liệu để train model

y = df['Classification'].values.reshape(-1,1)

X = df.drop(columns=['Classification'])
print(X.shape)
print(y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.1)
#Xay dung cay
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

my_tree = DecisionTreeClassifier(splitter='random')
my_tree.fit(X_train, y_train)
# Du doan tren du lieu test

y_pred = my_tree.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print(cm)
df = pd.read_csv("/content/datareg.csv")
df.head()

X = df['Cost'].values.reshape(-1,1)
y= df['Profit'].values.reshape(-1,1)
     
     