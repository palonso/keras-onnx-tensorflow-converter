from argparse import ArgumentParser

import onnx
from onnx import TensorProto
from onnx.helper import make_model, make_graph, make_tensor_value_info
from onnx.checker import check_model


def extract_subgraph(
    in_file,
    out_file,
    first_layer,
    last_layer,
    n_inputs,
    n_outputs,
    model_name="model",
):
    in_model = onnx.load(in_file)

    print("** input nodes **")
    for node in in_model.graph.node:
        print(
            "name=%r type=%r input=%r output=%r"
            % (node.name, node.op_type, node.input, node.output)
        )

    # select nodees
    selected_nodes = []
    for node in in_model.graph.node:
        if node.name == first_layer:
            selected_nodes.append(node)
            first_node = node
        elif len(selected_nodes) > 0:
            selected_nodes.append(node)
        if node.name == last_layer:
            last_node = node
            break

    x = make_tensor_value_info(first_node.input[0], TensorProto.FLOAT, ["batch_size", n_inputs])
    inputs = [x]
    y = make_tensor_value_info(last_node.output[0], TensorProto.FLOAT, ["batch_size", n_outputs])
    outputs = [y]

    required_inputs = set()
    for node in selected_nodes:
        for input_name in node.input:
            required_inputs.add(input_name)

    initializers = []
    for initializer in in_model.graph.initializer:
        if initializer.name in required_inputs:
            initializers.append(initializer)

    graph = make_graph(
        nodes=selected_nodes,
        name=model_name,
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    out_model = make_model(graph)

    print("** output nodes **")
    for node in out_model.graph.node:
        print(
            "name=%r type=%r input=%r output=%r"
            % (node.name, node.op_type, node.input, node.output)
        )

    # Let's check the model is consistent
    check_model(out_model)

    with open(out_file, "wb") as f:
        f.write(out_model.SerializeToString())

    print("done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("in_file", help="input ONNX file")
    parser.add_argument("out_file", help="output ONNX file")
    parser.add_argument("first_layer", help="first layer to convert", default=None)
    parser.add_argument("last_layer", help="last layer to convert", default=None)
    parser.add_argument("n_inputs", help="number of inputs", type=int)
    parser.add_argument("n_outputs", help="number of outputs", type=int)
    args = parser.parse_args()

    extract_subgraph(
        args.in_file,
        args.out_file,
        args.first_layer,
        args.last_layer,
        args.n_inputs,
        args.n_outputs,
    )
