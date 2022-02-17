from argparse import ArgumentParser
import tensorflow.compat.v1 as tf
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

def convert_pb_to_server_model(pb_model_path, export_dir, input_name='input:0', output_name='output:0'):
    graph_def = read_pb_model(pb_model_path)
    convert_pb_saved_model(graph_def, export_dir, input_name, output_name)


def read_pb_model(pb_model_path):
    with tf.gfile.GFile(pb_model_path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        return graph_def


def convert_pb_saved_model(pb_model_path, export_dir, input_name='input:0', output_name='output:0'):
    with tf.gfile.GFile(pb_model_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir)

    sigs = {}
    with tf.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def, name="")
        g = tf.get_default_graph()
        inp = g.get_tensor_by_name(input_name)
        out = g.get_tensor_by_name(output_name)

        sigs[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = \
            tf.saved_model.signature_def_utils.predict_signature_def(
                {"input": inp}, {"output": out})

        builder.add_meta_graph_and_variables(sess,
                                             [tag_constants.SERVING],
                                             signature_def_map=sigs)
        builder.save()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("input_name")
    parser.add_argument("output_name")
    args = parser.parse_args()

    convert_pb_saved_model(
        args.input,
        args.output,
        args.input_name + ":0",
        args.output_name + ":0",
    )
