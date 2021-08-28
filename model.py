import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle


df = pd.read_csv("Used_Bikes.csv")
df.drop(['bike_name'],axis=1,inplace=True)
df_copy = df.drop(["owner","brand","city"],axis = 1)
df_final = pd.concat([df_copy, pd.get_dummies(df.city), pd.get_dummies(df.owner), pd.get_dummies(df.brand)],axis = 1)

x = df_final.iloc[:,1:]
y = df_final.iloc[:,0]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state=2)

model = RandomForestRegressor()
model.fit(x_train,y_train)

pickle.dump(model, open("model.pkl","wb"))
