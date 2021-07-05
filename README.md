# IndividualLevel-PathSpecific-Counterfactual-Fairness

This repository is an official implementation of the following paper:

Yoichi Chikahara, Shinsaku Sakaue, Akinori Fujino, Hisashi Kashima; 'Learning Individually Fair Classifier with Path-Specific Causal-Effect Constraint', International Conference on Artificial Intelligence and Statistics (AISTATS2021).

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training and Evaluation

To train the classifier and evaluate its performance, run this command:

```train
./run.sh <penalty parameter value (e.g., 2.0)>
```

## Configurations

By modifying script file "run.sh", you can run the code with various settings:

- filename: You can select classifier, method, and data. For instance, with "filename=${i}_DNNProposed_synth", you choose to test our method with DNN classifier on synthetic data. The main options are follows:
-- classifier: "DNN" and "Logistic"
-- method: "Proposed", "FIO" (constraint on mean unfair effect), "Remove" (remove several input features), and "Unconstrained" (no fairness constraint)
-- data: "synth", "german", and "adult"

- T: The number of epochs
- lr, mom, batchsize, opt: Parameters are used in optimization (torch.optim)

## Reference

If you use this code, please cite the following paper:

@inproceedings{chikahara2021learning,
  title={Learning individually fair classifier with path-specific causal-effect constraint},
  author={Chikahara, Yoichi and Sakaue, Shinsaku and Fujino, Akinori and Kashima, Hisashi},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={145--153},
  year={2021},
  organization={PMLR}
}

## License

This repository is licensed under "license.txt"
