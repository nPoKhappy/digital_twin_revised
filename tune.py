import argparse, os, subprocess, uuid, yaml, pandas as pd, optuna
import sys
def run_train(cfg):
    exp = cfg['exp_name']
    tmp_dir = './tmp_configs'; os.makedirs(tmp_dir, exist_ok=True)
    tmp_cfg = os.path.join(tmp_dir, f'{exp}.yaml')
    with open(tmp_cfg, 'w', encoding='utf-8') as f: yaml.safe_dump(cfg, f, allow_unicode=True)
    subprocess.run([sys.executable, 'train.py', '--config', tmp_cfg], check=True)
    return exp

def read_valid_metric(exp):
    # 以驗證 MAE 作為目標（越小越好）
    p = os.path.join('results', exp, 'MAE_timestep_valid.csv')
    if not os.path.exists(p): return 1e9  # 若缺檔，給大罰分
    df = pd.read_csv(p)
    return pd.to_numeric(df.select_dtypes('number').values.flatten(), errors='coerce').mean()

def objective(trial, base_cfg):
    cfg = yaml.safe_load(open(base_cfg, 'r', encoding='utf-8'))
    base_exp = cfg['exp_name']

    # 搜索空間
    cfg['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)
    cfg['training']['batch_size'] = trial.suggest_categorical('batch_size', [256, 512, 1024])
    cfg['model']['embedding_dim'] = trial.suggest_categorical('embedding_dim', [32, 48, 64])
    cfg['model']['hidden_dim'] = trial.suggest_categorical('hidden_dim', [32, 48, 64])
    cfg['model']['n_layers'] = trial.suggest_int('n_layers', 2, 4)

    # 加速試驗（可依需求調整）
    cfg['training']['epochs'] = min(cfg['training'].get('epochs', 10000), 300)
    cfg['training']['patience'] = min(cfg['training'].get('patience', 100), 30)

    # 唯一實驗名，避免覆蓋
    cfg['exp_name'] = f"{base_exp}_opt_{trial.number}_{uuid.uuid4().hex[:6]}"

    try:
        exp = run_train(cfg)
        score = read_valid_metric(exp)  # 越小越好
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        score = 1e9
    return score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-config', default='configs/transformer_experiment_AT_step_test.yaml')
    ap.add_argument('--n-trials', type=int, default=20)
    args = ap.parse_args()

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda t: objective(t, args.base_config), n_trials=args.n_trials)
    print('Best value:', study.best_value)
    print('Best params:', study.best_params)

if __name__ == '__main__':
    main()