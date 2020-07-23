import argparse
import os
import onnx
from onnx_tf.backend import prepare


def o2pb(onnx_model=None, pb_model=None, force=False):
    """Converts an ONNX model into TensorFlow's Protocol Buffer (.pb) format."""
    
    if not os.path.exists(pb_model) or force:
        model = onnx.load(onnx_model)  # load onnx model

        tf_rep = prepare(model)  # prepare tf representation
        tf_rep.export_graph(pb_model)  # export the model
    else:
        print('"{}" already exists. Change `pb_model` name or use `force`.'.format(pb_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Converts an ONNX model into TensorFlow's Protocol Buffer (.pb) format.")
    parser.add_argument('onnx_model')
    parser.add_argument('pb_model')
    parser.add_argument('-f', '--force', action='store_true')

    o2pb(**vars(parser.parse_args()))
