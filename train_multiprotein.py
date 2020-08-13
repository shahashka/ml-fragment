import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import glob
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score, precision_score, auc, recall_score, roc_curve
from regression_enrichment_surface import regression_enrichment_surface as rds

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def r2_keras(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def load_data():
    with open('dset.pkl', 'rb') as pickle_file:
        x,y = pickle.load(pickle_file)
    return X, y


def create_small_model(shape):
    with strategy.scope():
        dr = 0.1
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(6, 6),
                         activation='relu',
                         input_shape=shape))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Dropout(dr))
        model.add(Flatten())
    #     model.add(Dense(128, activation='relu'))
    #     model.add(Dense(64, activation='relu'))
    #     model.add(Dropout(dr))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(dr))
        model.add(Dense(1))

        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=tf.keras.optimizers.Adam(lr=1e-5),
                      metrics=['mean_squared_error', r2_keras])
        print(model.summary())
    return model


def train_predict(images, scores, batch_size=64, num_epochs=50):
    scores = np.abs(scores['Chemgauss4'].to_numpy())
    scaler = MinMaxScaler()
    scores = scaler.fit_transform(scores.reshape(-1,1))
    X_train,X_test, y_train, y_test = train_test_split(images, scores, test_size=0.2, shuffle=True)
    
    # Distribution
    plt_train=scaler.inverse_transform(y_train.reshape(-1,1))
    plt_test=scaler.inverse_transform(y_test.reshape(-1,1))
    plt.hist(plt_train, alpha=0.5, label='train')
    plt.hist(plt_test,alpha=1, label='test')
    plt.legend()
    plt.savefig("temp/dist.png")
    plt.clf()

    
    # Training
    model = create_small_model(images[0].shape) 
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs,
                        verbose=1, batch_size=batch_size)
    
    # Prediction
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(y_pred)
    y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))

    print("r2", r2_score(y_test_inv, y_pred_inv))
    print("pearson", pearsonr(y_test_inv.flatten(),y_pred_inv.flatten()))
    print("MSE actual", mean_squared_error(y_test_inv, y_pred_inv))

    plt.hist(y_test_inv, alpha=0.5, label="test")
    plt.hist(y_pred_inv,alpha=0.5, label="pred")
    plt.legend()
    plt.figtext(0.5, -0.1,
                " r2 {0}\n pearson {1}\n MSE actual {2}".format(r2_score(y_test_inv, y_pred_inv),                                                           pearsonr(y_test_inv.flatten(),y_pred_inv.flatten()),
                            mean_squared_error(y_test_inv, y_pred_inv) ), ha="center", fontsize=12)

    plt.savefig("temp/predict.png")
    plt.clf()
    
    
    # Enrichment
    values = y_test_inv.flatten()
    preds = y_pred_inv.flatten()
    quant = np.quantile(values, 0.90)
    values_bin = [0 if val < quant else 1 for val in values]
    preds_bin = [0 if val < quant else 1 for val in preds]
    print("precision, recall (90% quant)" ,precision_score(values_bin, preds_bin), recall_score(values_bin, preds_bin))
    fpr, tpr, thresholds = roc_curve(values_bin, preds_bin)
    print("auc", auc(fpr, tpr))

    rds_model = rds.RegressionEnrichmentSurface(percent_min=-4)
    rds_model.compute(values, preds, samples=50)
    rds_model.plot(title='Regression Enrichment Surface contact matrix model')
    plt.savefig("temp/res.png")
    return history


def plot( history):
    # Plot training & validation accuracy values
    fig , (ax1,ax2) = plt.subplots(nrows=2)
    ax1.plot(history.history['r2_keras'])
    ax1.plot(history.history['val_r2_keras'])
    ax1.set_title('Model r2')
    ax1.set_ylabel('r2')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Test'], loc='upper left')
    plt.savefig("temp/metrics.png")
  
    
def main():
    images, scores = load_data()
    history = train_predict(images,scores, batch_size=64,num_epochs=50)
    plot(history)

if __name__ == "__main__":
    main()    
