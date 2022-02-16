from argparse import ArgumentParser
import struct
import onnx

def add_output_node(
    model_in=None,
    model_out=None,
    batch_size=None,
):
    model = onnx.load(model_in)
    model = onnx.shape_inference.infer_shapes(model)
    graph = model.graph

    # Change batch size in input, output and value_info
    for tensor in list(graph.input) + list(graph.value_info) + list(graph.output):
        tensor.type.tensor_type.shape.dim[0].dim_value = batch_size

    # Set dynamic batch size in reshapes (-1)
    for node in graph.node:
        if node.op_type != 'Reshape':
            continue
        for init in graph.initializer:
            # node.input[1] is expected to be a reshape
            if init.name != node.input[1]:
                continue
            # Shape is stored as a list of ints
            print(init.int64_data)
            if len(init.int64_data) > 0:
                # This overwrites bias nodes' reshape shape but should be fine
                init.int64_data[0] = -1
            # Shape is stored as bytes
            elif len(init.raw_data) > 0:
                shape = bytearray(init.raw_data)
                struct.pack_into('q', shape, 0, -1)
                init.raw_data = bytes(shape)

    onnx.save(model, model_out)



if __name__ == "__main__":
    parser = ArgumentParser(
        "Change model's batch size")
    parser.add_argument('model_in')
    parser.add_argument('model_out')
    parser.add_argument('batch_size', type=int)

    add_output_node(**vars(parser.parse_args()))
