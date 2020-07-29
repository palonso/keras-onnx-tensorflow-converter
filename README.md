# keras-onnx-tensorflow-converter
Wrapper scripts to convert Keras models to ONNX and Protocol Buffers (.pb)

## Install
We have different requirements files for TensorFlow 1.X and 2.X. So far, `requirements_TF1.txt` has only been tested for TensorFlow 1.15.2.

```ShellSession
python3 -m virtualenv venv
source venv/bin/activate
pip install -r requirements_TFX.txt
```

## Usage
`k2pb.py` is the main script in this repository. It takes a Keras (.h5) file as input and produces a TensorFlow Protocol Buffer (.pb) as output using ONNX as an intermediate exchange format. By setting the flag `--rename` or specifing `--input` or `--outout` the model's input and output can be renamed.

```ShellSession
python3 src/k2pb.py --help
usage: Converts a Keras model into TensorFlow's Protocol Buffer (.pb) format.
       [-h] [-f] [-r] [-i INPUT_NAME] [-o OUTPUT_NAME]
       keras_model pb_model

positional arguments:
  keras_model
  pb_model

optional arguments:
  -h, --help            show this help message and exit
  -f, --force
  -r, --rename
  -i INPUT_NAME, --input-name INPUT_NAME
  -o OUTPUT_NAME, --output-name OUTPUT_NAME
```

`k2o.pb` takes a Keras (.h5) file and writes an ONNX file.

```ShellSession
python3 src/k2o.py --help
usage: Converts a Keras model into ONNX format.
       [-h] [-f] keras_model onnx_model

positional arguments:
  keras_model
  onnx_model

optional arguments:
  -h, --help   show this help message and exit
  -f, --force
```

`o2pb.pb` takes an ONNX file and converts it into a TensorFlow's Protocol Buffer (.pb).

```ShellSession
python3 src/o2pb.py --help
usage: Converts an ONNX model into TensorFlow's Protocol Buffer (.pb) format. [-h] [-f] onnx_model pb_model

positional arguments:
  onnx_model
  pb_model

optional arguments:
  -h, --help   show this help message and exit
  -f, --force
```

`clean_node_names.py` is and auxiliary script to remove invalid characters according to TensorFlow's [scope naming convention](https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/framework/ops.py#L2993). Running this script may be a requirement before converting an ONNX model to TensorFlow's .pb format.

```ShellSession
python3 src/clean_node_names.py --help
usage: Removes characters from an ONNX model node names that are not matching a re `pattern`By default it removes characters unsupported by the TensorFlow scope names criteria:
https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/python/framework/ops.py#L2993
       [-h] [-f] [-p PATTERN] model_in model_out

positional arguments:
  model_in
  model_out

optional arguments:
  -h, --help            show this help message and exit
  -f, --force
  -p PATTERN, --pattern PATTERN

```

`change_interface_names.py` is an auxiliary script to change the input and output names from an ONNX model. Useful to make the models cleaner for deployment.
Note that this script is designed to operate on the first input and the first output of the model, the rest of the inputs and outputs are ignored.

```ShellSession
python3 src/change_interface_names.py --help
usage: Renames the first input and output node names of `model_in` to `input_name` and `output_name` and saves it to `model_out`.
       [-h] [-f] [-i INPUT_NAME] [-o OUTPUT_NAME] model_in model_out

positional arguments:
  model_in
  model_out

optional arguments:
  -h, --help            show this help message and exit
  -f, --force
  -i INPUT_NAME, --input-name INPUT_NAME
  -o OUTPUT_NAME, --output-name OUTPUT_NAME
```
