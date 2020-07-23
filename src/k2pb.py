import argparse
import os

from k2o import k2o
from clean_node_names import clean_node_names
from change_interface_names import change_interface_names
from o2pb import o2pb

TMP_MODEL = 'tmp.onnx'
INPUT_NAME = 'input'
OUTPUT_NAME = 'output'

def k2pb(keras_model=None, pb_model=None, force=False, rename=False, input_name=None, output_name=None):
    """Converts a Keras model into TensorFlow's Protocol Buffer (.pb) format."""

    if not os.path.exists(pb_model) or force:
        try:
            k2o(keras_model, TMP_MODEL, force=True)
            clean_node_names(TMP_MODEL, TMP_MODEL, force=True)

            if rename:
                if not input_name:
                    input_name = INPUT_NAME
                if not output_name:
                    output_name = OUTPUT_NAME

            names = dict()
            if input_name:
                names['input_name'] = input_name
            if output_name:
                    names['output_name'] = output_name

            if names:
                change_interface_names(TMP_MODEL, TMP_MODEL, force=True, **names)

            o2pb(TMP_MODEL, pb_model, force=force)
        finally:
            os.remove(TMP_MODEL)
    else:
        print('"{}" already exists. Change `pb_model` name or use `force`.'.format(pb_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Converts a Keras model into TensorFlow's Protocol Buffer (.pb) format.")
    parser.add_argument('keras_model')
    parser.add_argument('pb_model')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('-r', '--rename', action='store_true')
    parser.add_argument('-i', '--input-name')
    parser.add_argument('-o', '--output-name')

    k2pb(**vars(parser.parse_args()))
