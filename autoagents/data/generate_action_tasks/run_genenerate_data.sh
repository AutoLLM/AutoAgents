# Use this for chat API:
python -m generate_data_chat_api generate_agents_data \
  --output_dir ./new_data \
  --seed_tasks_path ./seed_tasks.jsonl \
  --num_agents_to_generate 1000 \
  --model_name="gpt-4"