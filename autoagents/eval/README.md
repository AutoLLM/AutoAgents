Use test.py to evaluate models:

```
PYTHONPATH=`pwd` python autoagents/eval/test.py --help
```
```
usage: test.py [-h] [--model MODEL] [--temperature TEMPERATURE] [--agent [{ddg,wiki}]] [--persist-logs]
               [--dataset [{default,hotpotqa,ft,hf,bamboogle}]] [--eval] [--prompt-version [{v2,v3}]] [--slice SLICE]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         model to be tested
  --temperature TEMPERATURE
                        model temperature
  --agent [{ddg,wiki}]  which action agent we want to interact with(default: ddg)
  --persist-logs        persist logs on disk, enable this feature for later eval purpose
  --dataset [{default,hotpotqa,ft,hf,bamboogle}]
                        which dataset we want to interact with(default: default)
  --eval                enable automatic eval
```

Sample command to eval on hotpotqa dataset:
```
PYTHONPATH=`pwd` python autoagents/eval/test.py --model gpt-4 --temperature 0 --agent wiki --persist-logs --dataset hotpotqa --eval
```

Sample command to eval on bamboogle dataset:
```
PYTHONPATH=`pwd` python autoagents/eval/test.py --model gpt-4 --temperature 0 --agent ddg --persist-logs --dataset bamboogle --eval
```
These commands will generate model logs under `data` folders automatically and run evaluation scripts on those logs.
