## Enhancing Code Representation With Additional Context

This repository contains the data and code for the journal paper "Enhancing Code Representation With Additional Context"

### Data 

Our data is published using Figshare, please download data from [here](https://figshare.com/s/71c3233d55c2ad91f30c) and put it into the folder data before running experiments.

## Structure
The structure of our source code's repository is as follows:
- Classification: contains our script experimenting Code Clone Detection and Code Classification Tasks;
- Summarization: contains our script experimenting Code Summarization Task;   
- env.yml: contains the configuration for our enviroment. 


### Replicating results in the Paper

To replicate our results, please follow the instruction as below:

- Code Clone Detection

```
bash experiment_clone.sh
```

- Code Classification

```
bash experiment_class.sh
```

- Code Summarisation

```
bash experiment_summarization.sh
```

### Supplementary Materials

#### Human Evaluation Data

- Human Evaluation result on Automated Code Summarisation (in human_evaluation.zip):
    - Pilot phases: `human_evaluation/Pilot Study/`
        
    - Main Study: `human_evaluation/Main Study/`

    - Example:
    
    ![An example for Human Evaluation task in Code Summarization with Rank-Order-with-Ties questions](_img/example_Q5.png)
    
## 📜 Citation

Please cite the following article if you find our research including findings, datasets and tools to be useful:

```
@misc{nguyen2025enhancingneuralcoderep,
      title={Enhancing Neural Code Representation With Additional Context}, 
      author={Nguyen, Huy and Treude, Christoph and Thongtanunam, Patanamon},
      year={2025},
      month=oct
      eprint={xxxx.xxxxx},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/xxxx.xxxxx}, 
}
```