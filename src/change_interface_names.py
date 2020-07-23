import argparse
import os
import onnx


def change_interface_names(model_in=None, model_out=None, input_name=None, output_name=None, force=False):
    """Renames the first input and output node names of `model_in` to `input_name` and `output_name` and saves it to `model_out`."""

    if not os.path.exists(model_out) or force:
        model = onnx.load(model_in)  # load onnx model

        nodes = model.graph.node
        
        if input_name:
            input_nodes = model.graph.input

            old_name = input_nodes[0].name
            input_nodes[0].name = input_name

            if len(input_nodes) > 1:
                print('WARNING: {} has {} input nodes, '
                    'but this script was designed for models with just one input.'.format(model_in, len(input_nodes)))

            for node in nodes:
                for i, node_input in enumerate(node.input):
                    if node_input == old_name:
                        node.input[i] = input_name

        if output_name:
            output_nodes = model.graph.output

            old_name = output_nodes[0].name
            output_nodes[0].name = output_name

            if len(output_nodes) > 1:
                print('WARNING: {} has {} output nodes, '
                    'but this script was designed for models with just one output.'.format(model_in, len(output_nodes)))

            for node in nodes:
                for i, node_output in enumerate(node.output):
                    if node_output == old_name:
                        node.output[i] = output_name

        onnx.save(model, model_out)
    else:
        print('"{}" already exists. Change `model_out` name or use `force`.'.format(model_out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        'Renames the first input and output node names of `model_in` to `input_name` and `output_name` and saves it to `model_out`.')
    parser.add_argument('model_in')
    parser.add_argument('model_out')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('-i', '--input-name')
    parser.add_argument('-o', '--output-name')

    change_interface_names(**vars(parser.parse_args()))
