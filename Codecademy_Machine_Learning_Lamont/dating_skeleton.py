# Author: Steven Lamont
# Title: Codecademy Machine Learning
# Date: 12 Nov 2018

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("profiles.csv")

# print(df.job.head())              #Print 5 headings for job
# print(df.smokes.value_counts())  #Print all choices and count for each

# Analysis of values and count by row
# Print(df.essay5.value_counts())

# Map into scores - template
# drink_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
# all_data["drinks_code"] = all_data.drinks.map(drink_mapping)

body_type_mapping = {"average": 3, "fit": 2, "athletic": 1, "thin": 2, "curvy": 4, "a little extra": 4,
                     "skinny": 2, "full figured": 5, "overweight": 5, "jacked": 1,
                     "used up": 4, "rather not say": 4}
df["body_type_index"] = df.body_type.map(body_type_mapping)
# print(df.body_type_index.value_counts())
# print(df['body_type_index'].head(20))

# print out histogram of body types
# plt.hist(df.body_type_index)
# plt.title("Body Type Histogram")
# plt.xlabel("Body Type")
# plt.ylabel("Count")
# plt.axis([1, 5, 0, 20000])
# #plt.xlim(16, 80)
# plt.show()

drinks_mapping = {"socially": 2, "rarely": 1, "often": 3, "not at all": 0, "very often": 4, "desperately": 5}
df["drinks_index"] = df.drinks.map(drinks_mapping)
# print("Drinks Mapping:")
# print(df.drinks_index.value_counts())
# print('Drinks')
# print(df['drinks_index'].head(20))
# print("")

smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 1, "yes": 2, "trying to quit": 1}
df["smokes_index"] = df.smokes.map(smokes_mapping)

education_mapping = {"graduated from college/university": 3, "graduated from masters program": 4,
                     "working on college/university": 2, "working on masters program": 3,
                        "graduated from two-year college": 2, "graduated from high school": 1,
                        "graduated from ph.d program": 6, "graduated from law school": 4,
                     "working on two-year college": 2, "dropped out of college/university": 2,
                     "working on ph.d program": 4, "college/university": 2,
                     "graduated from space camp": 2, "dropped out of space camp": 1,
                     "graduated from med school": 5, "working on space camp": 2,
                     "working on law school": 3, "two-year college": 2, "working on med school": 3,
                        "dropped out of two-year college": 2, "dropped out of masters program": 3,
                     "masters program": 4, "dropped out of ph.d program": 4,
                     "dropped out of high school": 0, "high school": 1,
                     "working on high school": 0, "space camp": 2,
                     "ph.d program": 4, "law school": 4, "dropped out of law school": 3,
                     "dropped out of med school": 3, "med school": 4}
df["education_index"] = df.education.map(education_mapping)
# print("Education Mapping:")
# print(df.education_index.value_counts())

diet_mapping = {"mostly anything": 1, "anything": 1,
                     "strictly anything": 1, "mostly vegetarian": 2,
                        "mostly other": 1, "strictly vegetarian": 3,
                        "vegetarian": 2, "strictly other": 1,
                     "mostly vegan": 2, "other": 1,
                     "strictly vegan": 3, "vegan": 3,
                     "mostly kosher": 1, "mostly halal": 1,
                     "strictly kosher": 2, "strictly halal": 2,
                     "halal": 1, "kosher": 1}
df["diet_index"] = df.diet.map(diet_mapping)
# print("Diet Mapping:")
# print(df.diet_index.value_counts())

# plt.scatter(df.drinks_index, df.body_type_index, alpha= 0.1)  Didn't show much correlation
# plt.show()

# Removing the NaNs
subarray = df[['body_type_index', 'diet_index']]
subarray = subarray[~np.isnan(subarray).any(axis=1)]

# x_train, x_test, y_train, y_test = train_test_split(subarray['education_index'], subarray['body_type_index'],
#                            train_size = 0.8, test_size = 0.2)

X = subarray[['diet_index']]
X = X.values.reshape(-1, 1)
Y = subarray[['body_type_index']]

regr = linear_model.LinearRegression()
regr.fit(X, Y)
# print("slope is: ",regr.coef_, "  intercept is: ", regr.intercept_)

plt.scatter(df.diet_index, df.body_type_index, alpha= 0.01)
plt.title('Diet versus Body Types')
plt.xlabel('Diet Level')
plt.ylabel('Body Type')

# Add the best fit line to the chart
X_best_fit = []
Y_best_fit = []
for i in range(1, 4):
    y_point = regr.intercept_[0] + (i * regr.coef_[0])
    X_best_fit.append(i)
    Y_best_fit.append(y_point[0])

plt.plot(X_best_fit, Y_best_fit, color = 'red', linestyle = '--')
# plt.show()

# Calculate the r squared
Y_predicted = regr.predict(X)
# print('R squared:  ', r2_score(Y, Y_predicted))

plt.close('all')

# Now let's try multilinear regression
sub_data = df[['body_type_index', 'education_index', 'drinks_index', 'smokes_index', 'diet_index']]

sub_data = sub_data[~np.isnan(sub_data).any(axis=1)]

X = sub_data[['education_index', 'drinks_index', 'smokes_index', 'diet_index']]
Y = sub_data[['body_type_index']]

lm = LinearRegression()

model = lm.fit(X, Y)

print("regression coefficients: ", model.coef_)

Y_predict = lm.predict(X)

plt.scatter(Y, Y_predict, alpha=0.01)
plt.title('Prediction Accuracy with MultiLinear Regression')
plt.xlabel('True Body Type')
plt.ylabel('Predicted Body Type')
plt.show()
print('R squared:  ', r2_score(Y, Y_predict))



