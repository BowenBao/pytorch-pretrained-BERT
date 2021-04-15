python examples/language-modeling/run_clm.py \
    --model_name_or_path microsoft/prophetnet-large-uncased \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm \
    --ortmodule \
