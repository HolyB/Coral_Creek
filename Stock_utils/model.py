from keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply
from keras.models import Model
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import numpy as np
import xgboost as xgb
import pandas as pd

# Updated functions for user's DataFrame
def attention_3d_block(inputs, single_attention_vector=False):
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_model(INPUT_DIMS, TIME_STEPS, lstm_units=64):
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))

    x = Conv1D(filters=64, kernel_size=1, activation='relu')(inputs)
    x = Dropout(0.3)(x)

    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(0.3)(lstm_out)
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    output = Dense(1, activation='sigmoid')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

def predict_with_data(data, data_yuan, name, modelname, INPUT_DIMS, TIME_STEPS):
    yindex = data.columns.get_loc(name)
    data = np.array(data, dtype='float64')
    data, normalize = NormalizeMult(data)
    data_y = data[:, yindex]
    data_y = data_y.reshape(data_y.shape[0], 1)

    testX, _ = create_dataset(data)
    _, testY = create_dataset(data_y)
    if len(testY.shape) == 1:
        testY = testY.reshape(-1, 1)

    model = attention_model(INPUT_DIMS, TIME_STEPS)
    model.load_weights(modelname)
    model.summary()
    y_hat = model.predict(testX)
    testY, y_hat = xgb_scheduler(data_yuan, y_hat)
    return y_hat, testY

def lstm_model(model_type, X_train, yuan_X_train):
    if model_type == 1:
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1)))
        model.add(Dense(units=1))
        yuan_model = Sequential()
        yuan_model.add(LSTM(units=50, activation='relu', input_shape=(yuan_X_train.shape[1], 5)))
        yuan_model.add(Dense(units=5))
    elif model_type == 2:
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50, activation='relu'))
        model.add(Dense(1))

        yuan_model = Sequential()
        yuan_model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(yuan_X_train.shape[1], 5)))
        yuan_model.add(LSTM(units=50, activation='relu'))
        yuan_model.add(Dense(5))
    elif model_type == 3:
        model = Sequential()
        model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(X_train.shape[1], 1)))
        model.add(Dense(1))

        yuan_model = Sequential()
        yuan_model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(yuan_X_train.shape[1], 5)))
        yuan_model.add(Dense(5))

    return model, yuan_model

def xgb_scheduler(data, y_hat):
    close = data.pop('close')
    data.insert(5, 'close', close)
    train, test = prepare_data(data, n_test=len(y_hat), n_in=6, n_out=1)
    testY, y_hat2 = walk_forward_validation(train, test)
    return testY, y_hat2

def xgboost_forecast(train, testX):
    train = np.asarray(train)
    trainX, trainy = train[:, :-1], train[:, -1]
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=20)
    model.fit(trainX, trainy)
    yhat = model.predict(np.asarray([testX]))
    return yhat[0]

def walk_forward_validation(train, test):
    predictions = list()
    train = train.values
    history = [x for x in train]
    for i in range(len(test)):
        testX, testy = test.iloc[i, :-1], test.iloc[i, -1]
        yhat = xgboost_forecast(history, testX)
        predictions.append(yhat)
        history.append(test.iloc[i, :])
        print(f"{i+1} > expected={testy:.6f}, predicted={yhat:.6f}")
    return test.iloc[:, -1], predictions