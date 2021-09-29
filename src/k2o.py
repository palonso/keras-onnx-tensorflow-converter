import argparse
import os
import tensorflow as tf

os.environ['TF_KERAS'] = '1'

from keras2onnx import convert_keras, save_model


def k2o(keras_model=None, onnx_model=None, force=False):
    """Converts a Keras model into ONNX format."""
    
    if not os.path.exists(onnx_model) or force:
        model = tf.keras.models.load_model(keras_model)
        save_model(convert_keras(model), onnx_model)
    else:
        print('"{}" already exists. Change `onnx_model` name or use `force`.'.format(onnx_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Converts a Keras model into ONNX format.')
    parser.add_argument('keras_model')
    parser.add_argument('onnx_model')
    parser.add_argument('-f', '--force', action='store_true')

    k2o(**vars(parser.parse_args()))
