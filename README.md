## Enhancing Code Representation With Additional Context

This repository contains the data and code for the journal paper "Enhancing Code Representation With Additional Context"

### Data 

Our data is published using Figshare, please download data from [here](https://figshare.com/s/71c3233d55c2ad91f30c). Please unzip and then put datasets to the following folder before running experiments.

- For Classification tasks (Code Clone Detection and Code Classification):  `Classification/data/` 

- For Code Summarisation tasks:  `Summarization/Task/Code-Summarization/dataset/java/` 

### Structure

The structure of our source code's repository is as follows:

- Classification: contains our script experimenting Code Clone Detection and Code Classification Tasks;

- Summarization: contains our script experimenting Code Summarization Task;   

- env.yml: contains the configuration for our enviroment. 

### Replicating results in the Paper

To replicate our results, please follow the instruction as below:

- Code Clone Detection: Please use the folder `Classification/`

```
bash experiment_clone.sh
```

- Code Classification: Please use the folder `Classification/`

```
bash experiment_class.sh
```

- Code Summarisation: we can use the respective model's folder (e.g. `CodeBERT`, `GraphCodeBERT`, `CodeT5`, and `PLBART`) in the project directory at `Summarization/Task/Code-Summarization/`. 

    - For instance, to replicate `CodeT5`'s result, we use the following bash commands for different scenarios:

        - Without Context (baseline):

        ```
        $cd Summarization/Task/Code-Summarization/codet5
        $bash ./run_exp.sh java baseline
        ```
        - Code + Version History:
        
        ```
        $cd Summarization/Task/Code-Summarization/codet5
        $bash ./run_exp.sh java code_vh
        ```

        - Code + Call Graph:
        
        ```
        $cd Summarization/Task/Code-Summarization/codet5
        $bash ./run_exp.sh java code_cg
        ```

        - Code + Version History + Method Age:
        
        ```
        $cd Summarization/Task/Code-Summarization/codet5
        $bash ./run_exp.sh java code_vh_nod
        ```

        - Code + Version History + Call Graph:
        
        ```
        $cd Summarization/Task/Code-Summarization/codet5
        $bash ./run_exp.sh java code_vh_cg
        ```

        - Code + Call Graph + Version History:
        
        ```
        $cd Summarization/Task/Code-Summarization/codet5
        $bash ./run_exp.sh java code_cg_vh
        ```

        - Code + Version History + Call Graph + Method Age:
        
        ```
        $cd Summarization/Task/Code-Summarization/codet5
        $bash ./run_exp.sh java code_vh_cg_nod
        ```

        - Code + Call Graph + Version History + Method Age:
        
        ```
        $cd Summarization/Task/Code-Summarization/codet5
        $bash ./run_exp.sh java code_cg_vh_nod
        ```

### Supplementary Materials

#### Human Evaluation Data

- Human Evaluation result on Automated Code Summarisation is provided at [figshare](https://figshare.com/s/71c3233d55c2ad91f30c)(please unzip the file human_evaluation.zip):

    - Pilot phases: `human_evaluation/Pilot Study/`
        
    - Main Study: `human_evaluation/Main Study/`

    - Evaluation Guidelines for annotators: `human_evaluation/SURVEY_INSTRUCTION.pdf` (also downloadable from this repository)

    - Example of a Rank-Order question design (with ties, blinded source) in Qualtrics:
    
    ![An example for Human Evaluation task in Code Summarization with Rank-Order-with-Ties questions](_img/example_Q5.png)
    
## 📜 Citation

Please cite the following article if you find our research including findings, datasets and tools to be useful:

```
@misc{nguyen2025enhancingneuralcoderep,
      title={Enhancing Neural Code Representation With Additional Context}, 
      author={Nguyen, Huy and Treude, Christoph and Thongtanunam, Patanamon},
      year={2025},
      month=oct
      eprint={2510.12082},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2510.12082}, 
}
```