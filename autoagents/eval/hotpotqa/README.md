# Search LLM Evaluation

## Overview

### Dataset
- [HotpotQA](https://hotpotqa.github.io/)
    - [Fullwiki Dev Set](http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json)

### Metrics
- Recall/Precision/F1
- Mean Reciprocal Rank (MRR)
- Accuracy
- Missing Rate

## Results

- Overall
    | Metric | LLM | 30 Samples | 100 Samples | 500 Samples |
    | --- | --- | --- | --- | --- |
    | LLM Accuracy | GPT-3.5-turbo | 0.4 | 0.35 | 0.328|
    | LLM Accuracy | GPT-4 | **0.7** | **0.55** | **0.414** |
    | Supporting facts recall | GPT-3.5-turbo | 0.3833 | 0.355 | 0.313 |
    | Supporting facts recall | GPT-4 | **0.5333** | **0.44** | **0.353** |
    | Max MRR | GPT-3.5-turbo | 0.2606 | 0.2862 | 0.2412 |
    | Max MRR | GPT-4 | **0.3531** | **0.3229** | **0.2740** |
    | First MRR | GPT-3.5-turbo | 0.2522 | 0.2799 | 0.2355 |
    | First MRR | GPT-4 | **0.3531** | **0.3204** | **0.2704** |
    | Last MRR | GPT-3.5-turbo | 0.2592 | 0.2743 | 0.2288 |
    | Last MRR | GPT-4 | **0.3481** | **0.3127** | **0.2663** |

- Only on output with final answers
    | Metric | LLM | 30 Samples | 100 Samples | 500 Samples |
    | --- | --- | --- | --- | --- |
    | LLM Accuracy | GPT-3.5-turbo | 0.48 | 0.4667 | 0.5031 |
    | LLM Accuracy | GPT-4 | **0.8077** | **0.7534** | **0.6635** |
    | Supporting facts recall | GPT-3.5-turbo | 0.46 | 0.4733 | 0.4801 |
    | Supporting facts recall | GPT-4 | **0.6154** | **0.6027** | **0.5657** |
    | Max MRR | GPT-3.5-turbo | 0.3127 | 0.3816 | 0.3700 |
    | Max MRR | GPT-4 | **0.4074** | **0.4424** | **0.4390** |
    | First MRR | GPT-3.5-turbo | 0.3027 | 0.3732 | 0.3611 |
    | First MRR | GPT-4 | **0.4074** | **0.4389** | **0.4333** |
    | Last MRR | GPT-3.5-turbo | 0.311 | 0.3657 | 0.3510 |
    | Last MRR | GPT-4 | **0.4016** | **0.4283** | **0.4267** |

- Error rate
    | Metric | LLM | 30 Samples | 100 Samples | 500 Samples |
    | --- | --- | --- | --- | --- |
    | Parsing error rate | GPT-3.5-turbo | 0.0667 | 0.01 | 0.06 |
    | Parsing error rate | GPT-4 | 0.0667 | **0** | **0.008** |
    | Missiong rate | GPT-3.5-turbo | 0.1667 | **0.25** | **0.348** |
    | Missiong rate | GPT-4 | **0.1333** | 0.27 | 0.376 |

