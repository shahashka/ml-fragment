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

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"])
print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))
def r2_keras(y_true, y_pred):
    from tensorflow.keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def load_data(num_samples=10000):
    arr = pickle.load(open("data_wcharge_100k.pkl",'rb'))
    scores = "adrp_adpr_A_cat_sorted.csv"
    scores = pd.read_csv(scores)
    scores = scores['Chemgauss4'].iloc[0:num_samples].to_numpy()
    images = np.stack(arr[0:num_samples],axis=0)
    print(images.shape)
    scaler = MinMaxScaler()
    scores = np.abs(scores)
    scaled_scores = scaler.fit_transform(scores.reshape(-1,1))
    return images, scaled_scores

def create_large_model():
    with strategy.scope():
            base_model = tf.keras.applications.ResNet101( weights="imagenet",
                                        input_shape= (64,64,3), include_top=False)
            global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
            dr0 = tf.keras.layers.Dropout(0.2)
            linear_layer1=tf.keras.layers.Dense(64)
            dr1 = tf.keras.layers.Dropout(0.2)
            linear_layer2=tf.keras.layers.Dense(32)
            dr2 = tf.keras.layers.Dropout(0.2)
            #linear_layer3=tf.keras.layers.Dense(16)
            #dr3 = tf.keras.layers.Dropout(0.2)
            prediction_layer = tf.keras.layers.Dense(1)
            model = tf.keras.Sequential([
                base_model,
                global_average_layer,
                #dr0,
                linear_layer1,
                #dr1,
                #linear_layer2,
                #dr2,
                #linear_layer3,
                #dr3,
                prediction_layer
            ])
            model.compile(loss=tf.keras.losses.mean_squared_error,
                         optimizer=tf.keras.optimizers.Adam(lr=1e-3),
                      metrics=['mean_squared_error', r2_keras])
    return model

def create_small_model():
    with strategy.scope():
        model = Sequential()
        model.add(Conv2D(8, kernel_size=(3, 3),
                         activation='relu',
                         input_shape=(64,64,3),strides=2))
        model.add(Conv2D(32, (3, 3), activation='relu', strides=2))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Conv2D(8, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu'))
        #model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(1))

        model.compile(loss=tf.keras.losses.mean_squared_error,
                      optimizer=tf.keras.optimizers.Adam(lr=1e-4),
                      metrics=['mean_squared_error', r2_keras])
    return model

def prepare_for_training(X,y, cache=True, shuffle_buffer_size=1000, batch_size=64):
    # This is a small dataset, only load it once, and keep it in memory.
    # use `.cache(filename)` to cache preprocessing work for datasets that don't
    # fit in memory.
    
    def generator():
        for i, j in zip(X, y):
            yield i, j

    ds = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32), 
                                        output_shapes=(tf.TensorShape((64, 64, 3)), tf.TensorShape((1, ))))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return ds

def train(images, scores, isSmall=True, batch_size=64, num_epochs=50):
    X_train,X_test, y_train, y_test = train_test_split(images, scores, test_size=0.2, shuffle=True)
    model = create_small_model() if isSmall else create_large_model()
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs,
                        verbose=1, batch_size=batch_size)
    return history
    
def plot(history):
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
    plt.savefig("metrics.png")
    
def main():
    images, scores = load_data(num_samples=70000)
    history = train(images,scores, isSmall=True, batch_size=64,num_epochs=25)
    plot(history)

if __name__ == "__main__":
    main()    
