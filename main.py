import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file = 'data.csv'
df = pd.read_csv(file, usecols=["nilai","jam"])
#Statisik:
'''plt.plot(df['nilai'], df['jam'], 'o')
plt.xlabel('Nilai')
plt.ylabel('Jam')
plt.show()'''
x = df.drop(columns=['nilai'])
y = df['nilai']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train)
y_pred = lin_reg.predict(x_test)
score = lin_reg.score(x_test, y_test)
print(f"Model Score: ",score)

kasus1 = lin_reg.predict([[3.2]])
print("Nilai : ", kasus1)