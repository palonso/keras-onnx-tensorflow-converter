import argparse
import os
import re
import onnx

PATTERN = '[A-Za-z0-9.][A-Za-z0-9_.\\-/]*'


def clean_node_names(model_in=None, model_out=None, pattern=PATTERN, force=False):
    """Removes characters from an ONNX model node names that are not matching a re `pattern`.
    By default it removes characters unsupported by the TensorFlow scope names criterium:
    https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/framework/ops.py#L2993"""

    if not os.path.exists(model_out) or force:
        model = onnx.load(model_in)  # load onnx model
        re_obj = re.compile(pattern)

        for node in model.graph.node:
            node.name = ''.join(re_obj.findall(node.name))

        onnx.save(model, model_out)
    else:
        print('"{}" already exists. Change `model_out` name or use `force`.'.format(model_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Removes characters from an ONNX model node names that are not matching a re `pattern`'
        'By default it removes characters unsupported by the TensorFlow scope names criterium:'
        'https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/framework/ops.py#L2993')
    parser.add_argument('model_in')
    parser.add_argument('model_out')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('-p', '--pattern', default=PATTERN)

    clean_node_names(**vars(parser.parse_args()))
