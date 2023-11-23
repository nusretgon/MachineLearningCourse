import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler 
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Gradient Descent Functions
# Gradient Descent  ⊃ Theta Update ve Cost Function
def compute_cost(X, y, theta):
    h = np.dot(X, theta)
    m = len(y)
    J = 1 / (2 * m) * np.sum(np.power((h - y), 2))
    cost_values.append(J)
    return J

def update_theta(X, y, theta, learning_rate):
    X = X.values
    m = len(y)
    h = np.dot(X, theta)
    errors = h - y

    for i in range(len(theta)):
        gradient = np.sum(errors * X[:, i]) / m # X[:, i] means X(i)
        theta[i] -= learning_rate * gradient

def gradient_descent(X,Y,theta,learning_rate):
    before_iteration = 1
    later_iteration = 0
    counter=1
    while before_iteration>later_iteration and before_iteration-later_iteration>0.0001:
        before_iteration = compute_cost(X,Y,theta)
        update_theta(X,Y,theta,learning_rate)
        later_iteration = compute_cost(X,Y,theta)
        print(f"After {counter} Iteration: ",later_iteration,"\n")
        counter+=1
    print("Iteration end in", counter-1, "th time.")
    return theta,counter

def predict(X_test, theta):
    predictions = np.dot(X_test,theta)
    return predictions

# Data Manipulation
cost_values = []
theta = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01])
test_data = pd.read_csv("testDATA.csv")
train_data = pd.read_csv("trainDATA.csv")

# Data Cleaning
train_data = train_data.drop_duplicates()
test_data = test_data.drop_duplicates()

def filter_data(df):
    fuel_filter = df["fuel"].isin(["LPG", "CNG", "Electric"])
    owner_filter = df["owner"] != "Test Drive Car"
    return df[fuel_filter & owner_filter]

def encode_features(df, columns, encoder):
    for column in columns:
        df[column] = encoder.fit_transform(df[column])
    return df

def scale_features(df, columns, scaler):
    for column in columns:
        df[column] = scaler.fit_transform(df[column].values.reshape(-1, 1))
    return df

train_data = filter_data(train_data)
test_data = filter_data(test_data)

categorical_columns = ["fuel", "seller_type", "transmission", "owner"]
encoder = LabelEncoder()
train_data = encode_features(train_data, categorical_columns, encoder)
test_data = encode_features(test_data, categorical_columns, encoder)

numerical_columns = ["year", "km_driven", "selling_price"]
scaler = StandardScaler()
train_data = scale_features(train_data, numerical_columns, scaler)
test_data = scale_features(test_data, numerical_columns, scaler)

feature_columns = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner"]
X = train_data[feature_columns]
Y = train_data["selling_price"]
X_test = test_data[feature_columns]
Y_test = test_data["selling_price"]

learning_rate=0.001
new_theta,counter = gradient_descent(X,Y,theta,learning_rate)
iteration = list(range(0,counter))
predictions = predict(X_test, new_theta)

predictions_df = pd.DataFrame({'predicted_selling_price': predictions})
result_df = pd.concat([test_data, predictions_df], axis=1)
result_df.to_excel('predicted_values.xlsx', index=False)
result_df.head()

mse = mean_squared_error(Y_test,predictions)
mae = mean_absolute_error(Y_test,predictions)
r2 = r2_score(Y_test,predictions)

print("Mean Squared Error: ",mse,"\nMean Absolute Error:", mae,"\nr2_score:",r2)
"""
plt.plot(cost_values)
plt.title("Learning Rate=" + str(learning_rate))
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.show()
"""

plt.scatter(Y_test,predictions)
plt.ylabel('Gerçek Değerler')
plt.xlabel('Tahminler')
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--')
plt.title('Gerçek Değerler vs Tahminler')
plt.show()

predictions = pd.DataFrame({'predicted_selling_price': predictions})
prediction_table = pd.concat([test_data, predictions], axis=1)
prediction_table.to_excel('predicted_values.xlsx', index=False)
prediction_table.head()


