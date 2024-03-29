python generate.py \
    --device_ids 0,1 \
    --CUDA_VISIBLE_DEVICES 0,1 \
    --batch_size 32 \
    --seed 42 \
    --num_workers 4 \
    --data_path ../../MMTG-ZH/MMTG-dev/datasets/new_data_rating/final_test_50.pkl \
    --model_path ./models/debug \
    --tokenizer_path ./vocab/vocab.txt \
    --temperature 1.1 \
    --topk 10 \
    --topp 0.7 \
    --repetition_penalty 1.5 \
    --n_samples 10 \
    --save_samples
    --save_samples_path res/test.txt