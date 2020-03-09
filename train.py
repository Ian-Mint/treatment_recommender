import numpy as np
import argparse
from tensorflow import keras

from data_loader import Data
from model import build_model

vasopressin_threshold = 0.02  # Values below this should be considered 0


def main(args):
    run_name = f'batch{args.batch_size}-epochs{args.epochs}-lstm{args.layers}x{args.width}-lookback{args.lookback}.{args.name}'

    data = Data(lookback=args.lookback, batch_size=args.batch_size)
    assert not np.isnan(data.features).any()

    n_features = data.train.features.shape[2]

    if args.load_model:
        model = keras.models.load_model(args.model_path)
    else:
        model = build_model(args.width, args.batch_size, args.lookback, n_features, args.layers,
                            output_threshold=vasopressin_threshold)

    checkpoint = keras.callbacks.ModelCheckpoint('models/model.tf', monitor='val_loss', verbose=0,
                                                 save_best_only=True, save_weights_only=False, mode='auto',
                                                 save_freq='epoch')
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
                      callbacks=[checkpoint, ],
                      )
        )
        print(f"Epoch {i}/{args.epochs}")
        model.reset_states()

    model.save(f'models/{run_name}.tf')


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

    # Load Model
    parser.add_argument('--load_model', type=bool, default=False, help='If true, load model from model_path')
    parser.add_argument('--model_path', type=str, default='models/model.tf', help='Load if load_model set to true')

    args = parser.parse_args()
    main(args)
