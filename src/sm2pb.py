import argparse
import os
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

def o2pb(sm_model=None, pb_model=None, force=False):
    """Converts an ONNX model into TensorFlow's Protocol Buffer (.pb) format."""
    
    if not os.path.exists(pb_model) or force:
        with tf.Graph().as_default() as g:
            with tf.compat.v1.Session() as sess:
                tf.compat.v1.saved_model.load(sess, ['serve'], sm_model)
                graph = tf.compat.v1.get_default_graph()

                tf.io.write_graph(
                    graph_or_graph_def=graph,
                    logdir=".",
                    name=pb_model,
                    as_text=False
                )
    else:
        print('"{}" already exists. Change `pb_model` name or use `force`.'.format(pb_model))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Converts a TensorFlow's SavedModel into a Protocol Buffer (.pb) format.")
    parser.add_argument('sm_model')
    parser.add_argument('pb_model')
    parser.add_argument('-f', '--force', action='store_true')

    o2pb(**vars(parser.parse_args()))
