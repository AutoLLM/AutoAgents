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

### general counts

- total_logs

  Total number of log json files evaluated

- total_steps

  Total number of steps in all log files evaluated

- total_rewrites

  Total number of rewrites triggered

- total_valid

  Number of valid log files. In most cases, a log is valid when it does not contain any errors.

- valid_steps

  Aggregated number of steps in valid log files.

- search_invoked

  Number of times `Tool_Search` or `Tool_Wikipedia` is invoked.

- notepad_invoked

  Number of times `Tool_Notepad` is invoked.

- Endwith_{action/tool}

  Number of times a conversation ends with a specific tool.

- visit_in_plan

  Number of plans that start with `Visit` 

- len_hist

  Aggregated length of history trace

- duplicate_actions

  Number of duplicate {action}+{action_inputs} pairs

- Finish_with_dups

  Number of duplicate `Tool_Finish`+{action_inputs} pairs

- average_answer_missing

  The ratio of times when the agent fails to produce a final answer for an input sample.

- average_steps

  Average number of steps in a conversation to reach the final answer.

- total_samples

  The total number of samples/goals evaluated.

- finished_samples

  The number of samples where the agent is able to call `Tool_Finish` for it.

### error counts

Count the number of times a specific pattern of error occurs in the error log.

- invalid_tools_error

  Check whether error log contains "Invalid tool requested by the model.".

- context_len_error

  Check whether error log contains "This model's maximum context length".

- dns_error

  Check whether error log contains "[Errno -3] Temporary failure in name resolution".

- parse_error

  Check whether error log contains "Could not parse LLM output:".

- rate_limit_error

  Check whether error log contains "Rate limit reached for ".

- connection_error

  Check whether error log contains "[Errno 111] Connection refused".

- other_error

  Any other kinds of uncaught exceptions will be marked as other_error.

### plan patterns

Occurrence of action sequences

- Tool_Search->Tool_Notepad

- Tool_Search->Tool_Search->Tool_Notepad

- Tool_Search->Tool_Search->Tool_Search->Tool_Notepad

- …

### histograms

- len_history_trace

  histogram of the lengths of history trace

- len_initial_plan

  histogram of the lengths of initial plans
  