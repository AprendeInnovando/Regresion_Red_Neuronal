import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
import matplotlib.pyplot as plt


def load_dataset():
    dataset_path = keras.utils.get_file("auto-mpg.data",
                                        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_data = pd.read_csv(dataset_path, names=column_names,
                           na_values="?", comment='\t',
                           sep=" ", skipinitialspace=True)
    data = raw_data.copy()

    return raw_data, data


def clean_data(data):
    # Eliminamos valores desconocidos
    data = data.dropna()

    origin = data.pop('Origin')

    data['USA'] = (origin == 1) * 1.0
    data['Europe'] = (origin == 2) * 1.0
    data['Japan'] = (origin == 3) * 1.0

    return data


def describe_data(train_dataset):
    # sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
    # plt.show()
    train_statistics = train_dataset.describe()
    train_statistics.pop("MPG")
    train_statistics = train_statistics.transpose()
    return train_statistics


def build_model():
    model_train = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model_train.compile(loss='mse',
                        optimizer=optimizer,
                        metrics=['mae', 'mse'])
    return model_train


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    pd.set_option('display.max_columns', 10)
    pd.options.mode.chained_assignment = None  # default='warn'
    # Descargar dataset
    raw_dataset, dataset = load_dataset()

    # Limpiamos los datos
    dataset = clean_data(dataset)

    # Separar los datos en train, test
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # Visualizamos los datos
    train_stats = describe_data(train_dataset)

    # Separamos los atributos de las etiquetas
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')


    # Normalizar los datos
    def norm(x):
        return (x - train_stats['mean']) / train_stats['std']


    normed_train_data = norm(train_dataset)
    normed_test_data = norm(test_dataset)

    # Construimos el modelo
    model = build_model()


    # Entrenamos el modelo
    # Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs):
            if epoch % 100 == 0: print('')
            print('.', end='')


    EPOCHS = 1000

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[early_stop, PrintDot()])

    # Visualizamos el progreso del entrenamiento
    # plot_history(history)
    print('')
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    print('')

    test_predictions = model.predict(normed_test_data).flatten()

    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()

    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()
