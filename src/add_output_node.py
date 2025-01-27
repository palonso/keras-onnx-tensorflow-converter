from argparse import ArgumentParser
import onnx


def try_parse_int(item):
    result = item
    try:
        result = int(item)
    except ValueError:
        pass

    return result


def add_output_node(
    model_in=None,
    model_out=None,
    node_in=None,
    node_out=None,
    output_shape=None,
    node_type="identity",
):
    model = onnx.load(model_in)
    print(f"Searchoing for node {node_in} in {model_in}")

    new_output = None
    for node in model.graph.node:
        if node.name == node_in:
            new_output = node
            print("output node found!")

    if not new_output:
        raise Exception(f"Node `{node_in}` not found in `{model_in}`")

    # Infer the shapes of each node.
    #  inferred_model = onnx.shape_inference.infer_shapes(model)
    #  for info in inferred_model.graph.value_info:
    #      if info.name == new_output.output[0]:
    #          shape_info = info.type.tensor_type.shape

    output_shape = [try_parse_int(x) for x in output_shape]

    identity_node = onnx.helper.make_node(
        node_type, inputs=[new_output.output[0]], outputs=[node_out], name=f"{node_type}_{node_out}",
    )
    model.graph.node.append(identity_node)

    output_value_info = onnx.helper.make_tensor_value_info(
        node_out,
        onnx.TensorProto.FLOAT,
        shape=output_shape,
    )
    model.graph.output.append(output_value_info)

    onnx.checker.check_model(model)

    onnx.save(model, model_out)


if __name__ == "__main__":
    parser = ArgumentParser(
        "Adds a new output `node_out` pointing to a given `input_node` through an Identity node. "
        "We recommend using a tool such as Netron to check the architecture and node name on `model_in`"
    )
    parser.add_argument("model_in")
    parser.add_argument("model_out")
    parser.add_argument("node_in")
    parser.add_argument("node_out")
    parser.add_argument("--node-type", default="Identity")
    parser.add_argument("--output-shape", nargs="+", type=str)

    add_output_node(**vars(parser.parse_args()))
