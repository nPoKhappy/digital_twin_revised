# src/utils.py (已修改為無 horizon 的版本)

import torch
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import os
from tqdm import tqdm

def generate_results(model, loader, device, config, mean, std, y_tags, de_mv_tags, prefix, set_name):
    print(f"開始評估 {set_name}...")
    model.eval()
    
    all_predictions_list = []
    all_targets_list = []

    with torch.no_grad():
        for en_input, de_inputs, targets in tqdm(loader, desc=f"Predicting on {set_name} set"):
            en_input = en_input.to(device)
            all_future_mvs = de_inputs.to(device)
            
            _, encoder_hiddens = model.encoder(en_input)
            all_preds_tensor = model.decoder(all_future_mvs, encoder_hiddens)
            
            all_predictions_list.append(all_preds_tensor.cpu().numpy())
            all_targets_list.append(targets.numpy())

    # 將 list of batches 合併成一個大的 numpy array
    y_pred_cov = np.concatenate(all_predictions_list, axis=0)
    y_true_cov = np.concatenate(all_targets_list, axis=0)
    
    # 反標準化
    y_mean_series = mean[y_tags]
    y_std_series = std[y_tags]
    y_pred_cov = y_pred_cov * y_std_series.values + y_mean_series.values
    y_true_cov = y_true_cov * y_std_series.values + y_mean_series.values
    
    # 開始繪圖和計算指標
    pred_len = config['window']['prediction_length']
    num_output = len(y_tags)
    
    for t in tqdm(range(pred_len), desc=f"為 {set_name} 繪圖並計算指標"):
        metrics = {'R2': [], 'MAPE': [], 'RMSE': [], 'MAE': []}
        
        for yi in range(num_output):
            true = y_true_cov[:, t, yi]
            pred = y_pred_cov[:, t, yi]

            metrics['RMSE'].append(sqrt(mean_squared_error(true, pred)))
            metrics['R2'].append(r2_score(true, pred))
            non_zero_mask = true != 0
            if np.any(non_zero_mask):
                metrics['MAPE'].append(mean_absolute_percentage_error(true[non_zero_mask], pred[non_zero_mask]))
            else:
                metrics['MAPE'].append(0.0)
            metrics['MAE'].append(mean_absolute_error(true, pred))

        for metric_name, values in metrics.items():
            df = pd.DataFrame([values], columns=y_tags, index=[f't+{(t + 1)}'])
            path = os.path.join(config['output']['results_dir'], prefix, f'{metric_name}_timestep_{set_name}.csv')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if t == 0:
                df.to_csv(path)
            else:
                df.to_csv(path, mode='a', header=False)

# one_shot_forecast 函數保持不變，它已經是我們需要的形式了
def one_shot_forecast(model, encoder_input_initial, decoder_inputs_future, device):
    model.eval()
    encoder_input = encoder_input_initial.to(device)
    decoder_inputs = decoder_inputs_future.to(device)
    with torch.no_grad():
        _, encoder_hiddens = model.encoder(encoder_input)
        predictions = model.decoder(decoder_inputs, encoder_hiddens)
    return predictions