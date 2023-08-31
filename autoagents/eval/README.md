Use test.py to evaluate models:

Sample command to eval on hotpotqa dataset:
```
PYTHONPATH=`pwd` python autoagents/eval/test.py --model <model_name>  --temperature 0 --agent wiki --persist-logs --dataset hotpotqa --prompt-version v2 --eval
```

Sample command to eval on bamboogle dataset:
```
PYTHONPATH=`pwd` python autoagents/eval/test.py --model <model_name> --temperature 0 --agent ddg --persist-logs --dataset bamboogle --prompt-version v2 --eval
```
These commands will generate model logs under `data` folders automatically and run evaluation scripts on those logs.

Sample advanced usage:
```
PYTHONPATH=`pwd` python autoagents/eval/test.py --model gpt-4  --temperature 0 --agent ddg --persist-logs --dataset bamboogle --prompt-version v2 --slice 1 --eval
```
Detailed configuration can be found in the main function of the test.py file.
