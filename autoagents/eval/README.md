Use test.py to evaluate models:

```
PYTHONPATH=`pwd` python eval/test.py --model gpt-4  --temperature 0 --agent ddg --persist-logs --dataset bamboogle --eval
```
This command will use gpt-4 as the model with 0 as the temperature, ddg as the search agent and bamboogle as the test dataset. It will also generate model logs under `data` folders automatically and run evaluation scripts on those logs.

More configuration can be found in the main function of the test.py file.