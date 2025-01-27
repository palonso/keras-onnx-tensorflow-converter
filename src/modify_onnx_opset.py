
from argparse import ArgumentParser
import onnx
from onnx import version_converter

def convert_onnx_version(
        model_in,
        model_out,
        opset,
        ir_version,
        model_version,
    ):
    """
    """
    # Load original model
    model = onnx.load(model_in)
    
    # Set IR and model versions
    model.ir_version = ir_version
    model.model_version = model_version
    
    # Convert to the target opset
    model = version_converter.convert_version(model, opset)

    # Save model if output path provided
    onnx.save(model, model_out)


# Example usage
if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("model_in")
    parser.add_argument("model_out")
    parser.add_argument("--opset", type=int)
    parser.add_argument("--ir-version", type=int)
    parser.add_argument("--model-version", type=int)

    args = parser.parse_args()

    convert_onnx_version(
        model_in=args.model_in,
        model_out=args.model_out,
        opset=args.opset,
        ir_version=args.ir_version,
        model_version=args.model_version,
    )
