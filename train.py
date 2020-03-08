import numpy as np
import keras
import pickle
import argparse

from data_loader import Data
from model import build_model


vasopressin_threshold = 0.02  # Values below this should be considered 0


def main(args):
    run_name = f'batch{args.batch_size}-epochs{args.epochs}-lstm{args.layers}x{args.width}.{args.name}'

    data = Data(lookback=args.lookback, batch_size=args.batch_size)
    assert not np.isnan(data.features).any()

    n_samples = data.train.features.shape[0]
    n_features = data.train.features.shape[2]
    model = build_model(args.width, args.batch_size, args.lookback, n_features, args.layers,
                        output_threshold=vasopressin_threshold)

    # TODO: figure out a way to use tensorboard
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=config.tensorboard_log_path, histogram_freq=1)
    checkpoint = keras.callbacks.callbacks.ModelCheckpoint('models/model', monitor='val_loss', verbose=0,
                                                           save_best_only=False, save_weights_only=False, mode='auto',
                                                           period=5)
    history = []
    for i in range(args.epochs):
        history.append(
            model.fit(x=data.train.features,
                      y=data.train.vasopressin,
                      batch_size=args.batch_size,
                      epochs=1,
                      validation_data=(data.validate.features, data.validate.vasopressin),
                      verbose=2,
                      shuffle=False,
                      callbacks=[checkpoint, ])
        )
        print(f"Epoch {i}/{args.epochs}")
        model.reset_states()

    model.save(f'models/{run_name}')
    with open(f'logs/{run_name}.history', 'wb') as f:
        pickle.dump(history, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Files
    parser.add_argument('--name', type=str, default='model', help='name to append to saved model')

    # Training
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')

    # Model
    parser.add_argument('--lookback', type=int, default=4, help='number of time steps to look back')
    parser.add_argument('--width', type=int, default=16, help='width of the LSTM layers')
    parser.add_argument('--layers', type=int, default=2, help='number of LSTM layers')

    args = parser.parse_args()
    main(args)
