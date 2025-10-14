## 紀錄依賴的更新
uv pip freeze > requirements 
# 開始訓練
## 把下面### 換成對應的yaml黨名
uv run python uv run  python train.py --config configs/###.yaml 
# 開始測試
uv run python uv run  python predict.py --config configs/experiments_name.yaml 







