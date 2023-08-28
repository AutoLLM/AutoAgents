## SERVING

To serve the models, run the following simultaniously (in multiple terminals or using background tasks)


```
bash autoagents/serve/controller.sh
```

```
bash autoagents/serve/openai_api.sh
```

```
MODEL_PATH=/some/path/to/your/model CONDENSE_RESCALE=1 bash autoagents/serve/model_worker.sh 
```

You may have multiple `model_worker.sh` instances. If you are using LongChat, set `CONDENSE_RESCALE` to be whatever scaling you are using (e.g. 4 or 8)

### Prompt V3 serving

1. Start the model server
```
python3 autoagents/serve/action_model_worker.py --model-path /path/to/model/checkpoint --controller http://localhost:21001 --port 31008 --worker http://localhost:31008
```

2. Start the completion API server, default address http://localhost:8004
```
python3 autoagents/serve/action_api_server.py
```
