from argparse import ArgumentParser
import onnx
import onnx.helper as helper
from onnx import TensorProto


def try_parse_int(item):
    result = item
    try:
        result = int(item)
    except ValueError:
        pass

    return result

def extract_last_n_layers(
    model_path,
    n, 
    output_path=None,
    input_node_name=None,
    input_shape=None,
    output_node_names=None,
):
    """
    Extract last N layers from an ONNX model
    
    Args:
        model_path (str): Path to input ONNX model
        n (int): Number of layers to extract from end
        output_path (str, optional): Path to save extracted model
    
    Returns:
        onnx.ModelProto: Extracted model
    """
    # Load original model
    model = onnx.load(model_path)

    ir_version = model.ir_version
    model_version = model.model_version
    opset_import = model.opset_import

    print("Input model specs:")
    print("  ir_version:", ir_version)
    print("  model_version:", model_version)
    print("  opset_import:", opset_import)
    
    print("Checking input model")
    onnx.checker.check_model(model)
    
    # Get node names in original graph
    node_names = [node.name for node in model.graph.node]
    
    # Select last N nodes
    last_n_nodes = node_names[-n:]

    print("nodes in the new model:")
    print(last_n_nodes)
    
    selected_nodes = [node for node in model.graph.node if node.name in last_n_nodes]

    # Collect required initializers (weights)
    print("\nrequired initializers:")
    required_initializers = []
    for node in selected_nodes:
        # Find initializers used by these nodes
        
        if not node.name:
            continue

        for input_name in node.input:
            matching_init = [init for init in model.graph.initializer if init.name == input_name]

            if matching_init:
                print(f"node name: {node.name}")
                print(f"input node: {input_name}")

                for init in matching_init:
                    print("  ", init.name)

                required_initializers.extend(matching_init)


    input_shape = [try_parse_int(x) for x in input_shape]
    
    inputs = [helper.make_tensor_value_info(
        input_node_name, 
        TensorProto.FLOAT, 
        input_shape,
    )]

    output_node_names = set(output_node_names)
    outputs = [output for output in model.graph.output if output.name in output_node_names]

    print("model outputs:")
    print(outputs)
    
    # Create a new model with selected nodes
    extracted_graph = helper.make_graph(
        [node for node in model.graph.node if node.name in last_n_nodes],
        'extracted_model',
        inputs,
        outputs,
        initializer=required_initializers,
    )
    
    extracted_model = helper.make_model(
        extracted_graph,
        opset_imports=opset_import,
    )

    extracted_model.ir_version = ir_version
    extracted_model.model_version = model_version

    # Rename duplicate initializers
    unique_initializers = set()
    i = 0
    for initializer in extracted_model.graph.initializer:
        if initializer.name in unique_initializers:
            # Generate a unique name
            new_name = f"{initializer.name}_{i}"
            print(f"Renaming initializer '{initializer.name}' to '{new_name}'")
            initializer.name = new_name
            i += 1
        unique_initializers.add(initializer.name)

    onnx.checker.check_model(extracted_model)
    
    # Save model if output path provided
    if output_path:
        onnx.save(extracted_model, output_path)
    
    return extracted_model

# Example usage
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("model_in")
    parser.add_argument("model_out")
    parser.add_argument("n_nodes", type=int)
    parser.add_argument("input_node_name")
    parser.add_argument("--input-shape", nargs="+")
    parser.add_argument("--output-node-names", nargs="+")

    args = parser.parse_args()

    extract_last_n_layers(
        args.model_in,
        n=args.n_nodes,
        output_path=args.model_out,
        input_node_name=args.input_node_name,
        input_shape=args.input_shape,
        output_node_names=args.output_node_names,
    )
