model=$1
seed=$2
# python LAMA/run_config.py data=id_subsample model=$model model.seed=$seed train=True
python LAMA/run_config.py data=ood_human_prompts model=$model model.seed=$seed
python LAMA/run_config.py data=ood_object_distribution model=$model model.seed=$seed
python LAMA/run_config.py data=ood_keyboard model=$model model.seed=$seed
