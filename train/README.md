## Fine Tuning

To train an action finetuned model from Llama 2, do the following:

1.  Ensure that your cluster has enough GPU RAM for your model. Modify the `--nproc_per_node` and `--nnodes` values in each of the `train/scripts/` files to match your topology. If you are using only a single machine, you may hardcode `$RANK` to 0. Otherwise, ensure that `--master_addr` is set to the address of the rank 0 worker.
2.  Modify `--data_path`, `--output_dir`, and `--model_name_or_path` to match the data and models you are using. Note that the output of `conv_finetuning.sh` and `longchat_conv_finetuning.sh` will be used as the model input for `action_finetuning.sh` and `longchat_action_finetuning.sh`. If using LongChat, make sure that `CONDENSE_RESCALE` is set to the right value.
3.  Run either `conv_finetuning.sh` or `longchat_conv_finetuning.sh` with the appropriate environment variables set (see step 1). You will need to run it from the root directory of this repository.
3.  Run either `action_finetuning.sh` or `longchat_action_finetuning.sh` with the appropriate environment variables set (see step 1). Make sure that if you use LongChat for the first finetuning, you use LongChat for the second. You will need to run it from the root directory of this repository.
