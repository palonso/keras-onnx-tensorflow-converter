import argparse
import os
import onnx
from onnx_tf.backend import prepare


def o2pb(onnx_model=None, sm_model=None, force=False):
    """Converts an ONNX model into TensorFlow's SavedModel format."""
    
    if not os.path.exists(sm_model) or force:
        model = onnx.load(onnx_model)  # load ONNX model

        tf_rep = prepare(model)  # prepare tf representation
        tf_rep.export_graph(sm_model)  # export the model
    else:
        print('"{}" already exists. Change `sm_model` name or use `force`.'.format(sm_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Converts an ONNX model into TensorFlow's Saved Model format.")
    parser.add_argument('onnx_model')
    parser.add_argument('sm_model')
    parser.add_argument('-f', '--force', action='store_true')

    o2pb(**vars(parser.parse_args()))
