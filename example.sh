python examples/seq2seq/run_seq2seq.py \
    --model_name_or_path t5-large \
    --do_train \
    --do_eval \
    --task summarization \
    --dataset_name xsum \
    --output_dir ~/tmp/tst-summarization \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --overwrite_output_dir \
    --predict_with_generate \
    --ortmodule \
    --onnx_large_model True
