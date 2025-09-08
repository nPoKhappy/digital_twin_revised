## Writing dependencies into txt file
uv pip freeze > requirements 
# to start training
## change experiment name into yaml file name
uv run python uv run  python train.py --config configs/experiments_name.yaml 
## to start inferencing
uv run python uv run  python predict.py --config configs/experiments_name.yaml 






