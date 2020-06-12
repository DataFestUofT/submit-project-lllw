import numpy as np
import pandas
import csv
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
...

KEYWORD = ["China","Trump","StayHome","Vaccine","Ventilator","ICU","Flight","Isolation","Border","SARS"]
FIRST_LAYER = 12
SECOND_LAYER = 8

# build aa vanilla neural network for factor significance analysis
def main():
    dataset = load_csv()
    col_num = dataset.shape[1]
    model = build_model(col_num - 1)   
    for i in range(col_num):
        X = np.concatenate((dataset[:, :i], dataset[:, i+1:]), axis=1)
        Y = dataset[:, i]
        model.fit(X, Y, epochs=40, batch_size=2, verbose=0)
        with open("../VNN/"+KEYWORD[i]+".csv", mode="w") as f:
            result_csv = csv.writer(f)
            result_csv.writerow(KEYWORD[0:i]+KEYWORD[i+1:])
            result = []
            for j in range(col_num-1):
                # calculate local gradient
                X_test_0 = [-0.5] * (col_num-1)
                X_test_0[j] = -0.5001
                X_test_1 = [-0.5] * (col_num-1)
                X_test_1[j] = -0.4999
                pred = (model.predict(np.array([X_test_1]))[0,0] - model.predict(np.array([X_test_0]))[0,0]) / 0.0002
                result.append(pred)
            result_csv.writerow(result)
                



def build_model(input_dim):
    model = Sequential()
    model.add(Dense(FIRST_LAYER, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    model.add(Dense(SECOND_LAYER, activation="relu"))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model



def load_csv():
    csv_array = pandas.read_csv("../Final_Average.csv")
    # remove date and overall
    values = csv_array.values[:,1:-1]
    values = np.where(pandas.isnull(values), 0, values)
    return values



if __name__ == "__main__":
    main()