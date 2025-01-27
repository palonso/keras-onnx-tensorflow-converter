from argparse import ArgumentParser
import onnx
import onnx.helper as helper
from onnx import TensorProto
from onnx import version_converter

def extract_last_n_layers(model_path, n, output_path=None):
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
    
    # Get node names in original graph
    node_names = [node.name for node in model.graph.node]
    
    # Select last N nodes
    last_n_nodes = node_names[-n:]

    print("nodes in the new model:")
    print(last_n_nodes)
    
    selected_nodes = [node for node in model.graph.node if node.name in last_n_nodes]

    # Collect required initializers (weights)
    print("Required initializers:")
    required_initializers = []
    for node in selected_nodes:
        # Find initializers used by these nodes
        
        if not node.name:
            continue


        for input_name in node.input:
            print(f"input node: {input_name}")
            matching_init = [init for init in model.graph.initializer if init.name == input_name]
            required_initializers.extend(matching_init)

            for init in matching_init:
                print(init.name)

    print("\n")

    # Create a new graph with only these nodes
    inputs = [inp for inp in model.graph.input]

    new_input_name = "layer_11_embeddings"
    new_input_shape = tuple(["batch_size", 281, 768])
    
    inputs = [helper.make_tensor_value_info(
        new_input_name, 
        TensorProto.FLOAT, 
        new_input_shape,
    )]

    outputs = [
        model.graph.output[0],
        model.graph.output[1],
    ]

    print("outputs")
    print(outputs)
    
    # Create a new model with selected nodes
    extracted_graph = helper.make_graph(
        [node for node in model.graph.node if node.name in last_n_nodes],
        'extracted_model',
        inputs,
        outputs,
        initializer=required_initializers,
    )
    
    extracted_model = helper.make_model(extracted_graph)

    
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

    args = parser.parse_args()

    extract_last_n_layers(args.model_in, n=args.n_nodes, output_path=args.model_out)
