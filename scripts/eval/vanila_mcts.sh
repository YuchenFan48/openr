python reason/evaluation/evaluate.py \
    --LM Qwen2.5-7B-Instruct \
    --RM Qwen2.5-Math-PRM-7B  \
    --task_name MATH \
    --temperature 0.7 \
    --max_new_tokens 2048 \
    --num_sequence 8 \
    --tree_max_width 4 \
    --tree_max_depth 50 \
    --save_dir debug \
    --method vanila_mcts \
    --num_worker 32 \
    --controller_addr http://0.0.0.0:28777 \
    --local

# math-shepherd-mistral-7b-prm