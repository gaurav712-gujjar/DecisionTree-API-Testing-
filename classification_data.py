import pandas as pd 


df = pd.read_csv("Social_Network_Ads - Social_Network_Ads.csv")
# print(df.head(2))

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

df['Gender'] = le.fit_transform(df['Gender'])

# print(df.head(2))

from sklearn.model_selection import train_test_split

x = df.drop(columns = ['Purchased'])
y = df['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


from sklearn.tree import DecisionTreeRegressor

dt = DecisionTreeRegressor()

dt.fit(x_train, y_train)

y_pred = dt.predict(x_test)

from sklearn.metrics import accuracy_score

# print(accuracy_score(y_test, y_pred) )


import joblib
# save model 
joblib.dump(dt, 'dt_model.pkl')
print("model save")


