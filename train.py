import numpy as np
import keras
import pickle
import argparse

from data_loader import Data
from model import build_model
import config


def create_dataset(d: np.ndarray, look_back: int = 1):
    assert d.ndim == 3

    d = d.reshape(d.shape[0] * d.shape[1])
    x, y = [], []
    for i in range(len(d) - look_back - 1):
        a = d[i:(i + look_back), 0]
        x.append(a)
        y.append(d[i + look_back, 0])
    return np.array(x), np.array(y)


def main(args):
    data = Data()
    batch_size = 11  # other prime factors of len(data.hadm_id)==2068 are 4, 47

    n_samples = data.train.features.shape[0]
    n_features = data.train.features.shape[2]
    model = build_model(args.width, batch_size, data.maxlen, n_features)

    assert np.isnan(data.features).any()

    run_name = f'{args.run_name}-epochs{args.epochs}-batch{args.batch_size}'
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=config.tensorboard_log_path, histogram_freq=1)
    checkpoint = keras.callbacks.callbacks.ModelCheckpoint(f'models/{run_name}', monitor='val_loss', verbose=0,
                                                           save_best_only=True, save_weights_only=False, mode='auto',
                                                           period=5)
    history = model.fit(x=data.train.features,
                        y=data.train.vasopressin.reshape(n_samples, data.maxlen, 1),
                        batch_size=args.batch_size,
                        epochs=args.epochs,
                        validation_split=0.1,
                        verbose=2,
                        shuffle=False,
                        callbacks=[checkpoint, ])

    with open(f'logs/{run_name}.history') as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Files
    parser.add_argument('--run_name', type=str, default='model', help='path for saving trained models')

    # Training
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs')

    # Model
    parser.add_argument('--lookback', type=int, default=-1, help='number of time steps to look back')
    parser.add_argument('--width', type=int, default=16, help='width of the LSTM layer')

    args = parser.parse_args()
    main(args)
