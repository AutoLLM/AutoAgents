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
  --log-save-dir LOG_SAVE_DIR
                        dir to save logs
  --dataset [{default,hotpotqa,ft,hf,bamboogle}]
                        which dataset we want to interact with(default: default)
  --eval                enable automatic eval
  --prompt-version [{v2,v3}]
                        which version of prompt to use(default: v2)
  --slice SLICE         slice the dataset from left, question list will start from index 0 to slice - 1
```
Sample command to eval on Hotpotqa dataset:
```
PYTHONPATH=`pwd` python autoagents/eval/test.py --model gpt-4 --temperature 0 --agent wiki --persist-logs --dataset hotpotqa --prompt-version v2 --eval
```

Sample command to eval on Bamboogle dataset:
```
PYTHONPATH=`pwd` python autoagents/eval/test.py --model gpt-4 --temperature 0 --agent ddg --persist-logs --dataset bamboogle --prompt-version v2 --eval
```
These commands will generate model logs under `data` folders automatically and run evaluation scripts on those logs.


## Common Metrics
### errors
- invalid_tools_error

  Check whether error log contains "Invalid tool requested by the model.".

- context_len_error

  Check whether error log contains "This model's maximum context length is".

- dns_error

  Check whether error log contains "[Errno -3] Temporary failure in name resolution".

- parse_error

  Check whether error log contains "Could not parse LLM output:".

### stats
- average_search_invoked

  Average number of times Tool_Search or Tool_Wikipedia is invoked.

- average_notepad_invoked

  Average number of times Tool_Notepad is invoked.

- average_rewritten

  Average number of rewrites triggered.

- average_answer_missing

  The ratio of times when the agent fails to produce a final answer for an input sample.

- average_steps

  Average number of steps in a conversation to reach the final answer.

- total_samples

  The total number of samples/goals evaluated.

- finished_samples

  Th number of samples where the agent is able to call Tool_Finish for it.

### counters
- Endwith_{action}

  Occurrence of action sequences

  - Tool_Search->Tool_Notepad

  - Tool_Search->Tool_Search->Tool_Notepad

  - Tool_Search->Tool_Search->Tool_Search->Tool_Notepad

  - …

- visit_in_plan

- duplicate_actions

  number of duplicate action+action_inputs pairs

- Finish_with_dups

  number of duplicate Tool_Finish+action_inputs pairs

- len_history_trace

  histogram of the lengths of history trace

- len_initial_plan

  histogram of the lengths of initial plan
