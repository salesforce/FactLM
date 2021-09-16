# Creating the Environment
To create the environment, run
```
conda env create --file enviornment.yaml
```

And before running any code, you have to add `src` to your pythonpath and activate the environment, which you can do with:
```
source init.sh
```

# Running Experiments
Models can be found in `configs/models`
Data can be found in `configs/data`

To run a model on a single dataset, run:
```
python src/run_config.py model=<model_config_name> data=<data_config_name> [train=True] {...}
```
Include `train=True` to train the model on the dataset, otherwise the model will be loaded and evaluated. If there is no trained model a new one will be initialized and evaluated (which is likely undesirable!). All other config options can be overriden at the command line. Model options should be prefixed with "model." and data options hould be prefixed with "data.".

To run a model on all datasets, run:
```
python scripts/run_experiments.py model=<model_config_name> data=<data_config_name> [train=True] {...}
```


For example:
```
python scripts/run_experiments.py model=adaptive-prompt_prefix-lstm_bert-base-cased model.seed=37 train=True
```
Will train a prefix-lstm model with BERT-base-cased used as the large language model. It will train on the `id_subsample` dataset, and evaluate on the `ood` datasets.


Mixture of Experts and Oracle are a bit different because their configs specify paths to their component models--relation classifier (relclf) and ptuning. Do not train these. An example for MOE is:
```
python scripts/run_experiments.py model=moe_roberta-large model.relclf.model_path=configs/model/relclf_bert-base-cased-sd36.yaml model.relclf.data_path=configs/data/id_subsample model.ptuning.model_path=configs/model/p-tuning_sd36_roberta-large.yaml
```
And oracle only needs the ptuning path
```
python scripts/run_experiments.py model=oracle_roberta-large model.ptuning.model_path=configs/model/p-tuning_sd36_roberta-large.yaml
```

For training the P-tuning embeddings, we only use the subject, object pairs, not the filled-in templates, so to greatly speed up training, we use just the relations from the LAMA dataset, so each pair is used only once per epoch:
```
python src/run_config.py data=id model=p-tuning_bert-base-cased train=True data.train.template_path=data/templates/relations_lama.json data.dev.template_path=data/templates/relations_lama.json data.test.template_path=data/templates/relations_lama.json
```



# References
This repo is built off the repo found [here](https://github.com/THUDM/P-tuning/tree/main)

