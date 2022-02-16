from argparse import ArgumentParser
import onnx

def add_output_node(
    model_in=None,
    model_out=None,
    node_in=None,
    node_out=None,
):
    model = onnx.load(model_in)

    for node in model.graph.node:
        if node.name == node_in:
            new_output = node

    # Infer the shapes of each node.
    inferred_model = onnx.shape_inference.infer_shapes(model)
    for info in inferred_model.graph.value_info:
        if info.name == new_output.output[0]:
            shape_info = info.type.tensor_type.shape
            shape = [dim.dim_value for dim in shape_info.dim]

    identity_node = onnx.helper.make_node(
        'Identity',
        inputs=[new_output.output[0]],
        outputs=[node_out]
    )
    model.graph.node.append(identity_node)

    output_value_info = onnx.helper.make_tensor_value_info(
        node_out,
        onnx.TensorProto.FLOAT, shape=shape,
    )
    model.graph.output.append(output_value_info)

    onnx.save(model, model_out)



if __name__ == "__main__":
    parser = ArgumentParser(
        "Adds a new output `node_out` pointing to a given `input_node` through an Identity node."
        "We recommend using a tool such as Netron to check the architecture and node name on `model_in`")
    parser.add_argument('model_in')
    parser.add_argument('model_out')
    parser.add_argument('node_in')
    parser.add_argument('node_out')

    add_output_node(**vars(parser.parse_args()))
