import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import os

epochs = 20
batch_size = 4 # 64 
neurons = 60
dropout_rate = 0.4

def load_data():
    # load training data
    x_train = pickle.load(open("../datasets/100hz/x_train.pkl", "rb"))
    y_train = pickle.load(open("../datasets/100hz/y_train.pkl", "rb"))
    
    # load validation data
    x_validation = pickle.load(open("../datasets/100hz/x_validation.pkl", "rb"))
    y_validation = pickle.load(open("../datasets/100hz/y_validation.pkl", "rb"))

    # Combine train and val data
    x_train = np.concatenate([x_train, x_validation])
    y_train = np.concatenate([y_train, y_validation])

    # load test data
    x_test = pickle.load(open("../datasets/100hz/x_test.pkl", "rb"))
    y_test = pickle.load(open("../datasets/100hz/y_test.pkl", "rb"))


    return x_train, y_train, x_test, y_test

    

def evaluate_metrics(x_test, y_test):
    y_pred = model.predict(x_test).flatten()
    
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for index, tup in enumerate(zip(y_test, y_pred)):
        actual = tup[0]
        pred = tup[1]

        if pred >= 0.5:
            if actual == 1:
                TP += 1
            else:
                FP += 1
        else:
            if actual == 0:
                TN += 1
            else:
                FN += 1


    print("TP:", TP)
    print("TN:", TN)
    print("FN:", FN)
    print("FP:", FP)
    print("Sn:", (TP / (TP + FN)))
    print("Sp:", (TN / (TN + FP)))
    print("p0:", (TN / (TN + FN)))
    print("p1:", (TP / (TP + FP)))
    print("accuracy:", ((TP+TN) / (TP + TN + FP + FN)))


def plot_accuracy(history, metric="accuracy"):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("Model Accuracy - LSTM")
    plt.ylabel("Accuracy", fontsize="large")
    plt.xlabel("Total Number of Epochs", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()

def plot_loss(history, metric="loss"):
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("Model Loss - LSTM")
    plt.ylabel("Loss", fontsize="large")
    plt.xlabel("Total Number of Epochs", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()
    plt.close()


@tf.autograph.experimental.do_not_convert
def lstm_model(input_shape):

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.LSTM(neurons, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(dropout_rate)) # avoids overfitting


    # output = 1, binary classification
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    print(model.summary())

    model.compile(
        optimizer='adam',
        loss="binary_crossentropy", # used for binary classification
        metrics=['accuracy'], 
    )

     # train model
    callbacks = [
        # save best model
        tf.keras.callbacks.ModelCheckpoint(
            "best_lstm_model.h5", save_best_only=True, monitor="val_accuracy"
        )
        #tf.keras.callbacks.EarlyStopping(monitor='val_loss')
    ]

    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split = 0.2,
        verbose=1, # how to show progress bar
    )

    return model, history


if __name__ == '__main__':
    # to disable run time warnings
    tf.autograph.set_verbosity(0)

    # load data
    x_train, y_train, x_test, y_test = load_data()

    # create model - input_shape = (61, 58)
    model, history = lstm_model(input_shape=(x_train.shape[1:]))
   
    # test model
    train_loss, train_acc = model.evaluate(x_train, y_train)
    test_loss, test_acc = model.evaluate(x_test, y_test)


    # evaluation metrics
    evaluate_metrics(x_test, y_test)

    # plot metrics
    plot_loss(history)
    plot_accuracy(history)








