Use test.py to evaluate models:

Sample command to eval on bamboogle dataset:
```
PYTHONPATH=`pwd` python autoagents/eval/test.py --model gpt-4  --temperature 0 --agent ddg --persist-logs --dataset bamboogle --eval
```
This will use gpt-4 as the model with 0 as the temperature, ddg as the search agent and bamboogle as the test dataset. It will also generate model logs under `data` folders automatically and run evaluation scripts on those logs.

Sample command to eval on hotpotqa dataset:
```
python test.py --model <model_name>  --temperature 0 --agent wiki --persist-logs --dataset hotpotqa --eval
```

More configuration can be found in the main function of the test.py file.
