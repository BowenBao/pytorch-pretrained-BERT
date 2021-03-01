python examples/seq2seq/run_seq2seq.py \
    --model_name_or_path microsoft/prophetnet-large-uncased \
    --do_train \
    --do_eval \
    --task summarization \
    --dataset_name cnn_dailymail \
    --dataset_config_name 3.0.0 \
    --output_dir ~/tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --max_source_length 512 \
    # --ortmodule \
    # --onnx_large_model False
