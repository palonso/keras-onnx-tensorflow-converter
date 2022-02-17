# Recipe to freeze the EffnetDiscogs models
set -e


input_path=model.onnx
intermediate_path=tmp.onnx
batch_size=64
embeddings_layer=Flatten_238

sm_model=effnet_tmp_bs${batch_size}
pb_model=${sm_model}.pb

echo "adding new output node..."
python3 ../add_output_node.py ${input_path} ${intermediate_path} ${embeddings_layer} embeddings

echo "changing the batch size..."
python3 ../change_batch_size.py ${intermediate_path} ${intermediate_path} ${batch_size}

echo "saving onnx model as pb..."
python3 ../o2sm.py ${intermediate_path} ${sm_model} --force

python3 ../sm2pb.py ${sm_model} ${pb_model}

echo "done!"
