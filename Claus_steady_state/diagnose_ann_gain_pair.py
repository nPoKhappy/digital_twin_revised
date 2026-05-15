import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FULL_MAPPING = {
    'B34.SPo.SPo': 'acidgas_Fm', 'B17.PV.PV': 'air', 'S20.P.P': 'HEATER1_output_P', 'B33.SPo.SPo': 'air2_SP',
    'B17.SPo.SPo': 'air_SP', 'B35.SPo.SPo': 'COG_SP', 'AIR2.Fv.Fv': 'second_air2', 'S4.Fv.Fv': 'COG',
    'B18.SPo.SPo': 'burner_input_T_SP', 'B18.PV.PV': 'burner_input_T_PV', 'B19.SPo.SPo': 'burner_output_T_SP',
    'B19.PV.PV': 'burner_output_T_PV', 'BURNER_PC.SPo.SPo': 'burner_output_P_SP', 'BURNER_PC.PV.PV': 'burner_output_P_PV',
    'FURANCE_PC.SPo.SPo': 'fur_outputP_SP', 'FURANCE_PC.PV.PV': 'fur_outputP_PV', 'FURANCE.T.0.(0)': 'fur_inputT',
    'FURANCE.T.1.(1)': 'fur_temp', 'SEP1_PC.SPo.SPo': 'SEP1_P_SP', 'SEP1_PC.PV.PV': 'SEP1_P_PV', 'SEP1.T.T': 'SEP1_T',
    'SEP2_PC.SPo.SPo': 'SEP2_P_SP', 'SEP2_PC.PV.PV': 'SEP2_P_PV', 'SEP2.T.T': 'SEP2_T', 'SEP3_PC.SPo.SPo': 'SEP3_P_SP',
    'SEP3_PC.PV.PV': 'SEP3_P_PV', 'SEP3.T.T': 'SEP3_T', 'B21.SPo.SPo': 'HEATER1_output_T_SP',
    'B21.PV.PV': 'HEATER1_output_T_PV', 'B20.SPo.SPo': 'HEATER2_output_T_SP', 'B20.PV.PV': 'HEATER2_output_T_PV',
    'CAT1_PC.SPo.SPo': 'cat1_output_P_SP', 'CAT1_PC.PV.PV': 'cat1_output_P_PV', 'CAT2_PC.SPo.SPo': 'cat2_output_P_SP',
    'CAT2_PC.PV.PV': 'cat2_output_P_PV', 'S12.F.F': 'fur_F', 'S12.P.P': 'fur_inputP', 'S15.T.T': 'fur_outputT',
    'S16.F.F': 'WHB_F', 'S16.P.P': 'WHB_inputP', 'S16.T.T': 'WHB_inputT', 'S13.T.T': 'WHB_outputT',
    'S13.P.P': 'WHB_outputP', 'S36.F.F': 'HEATER1_F', 'S36.P.P': 'HEATER1_input_P', 'S36.T.T': 'HEATER1_input_T',
    'S21.F.F': 'cat1_F', 'S21.P.P': 'cat1_input_P', 'S21.T.T': 'cat1_input_temp', 'S22.T.T': 'cat1_output_temp',
    'S25.F.F': 'HEATER2_F', 'S25.P.P': 'HEATER2_input_P', 'S25.T.T': 'HEATER2_input_T', 'S27.F.F': 'cat2_F',
    'S27.P.P': 'cat2_input_P', 'S27.T.T': 'cat2_input_temp', 'S28.T.T': 'cat2_output_temp', 'S14.F.F': 'SEP1_F',
    'S23.F.F': 'SEP2_F', 'S29.F.F': 'SEP3_F', 'ACIDGAS.T.T': 'acidgas_T', 'ACIDGAS.P.P': 'acidgas_P',
    'ACIDGAS.Fcn.H2O.("H2O")': 'acidgas_H2O', 'ACIDGAS.Fcn.H2S.("H2S")': 'acidgas_H2S',
    'ACIDGAS.Fcn.CO2.("CO2")': 'acidgas_CO2', 'S33.Zn.SO2.("SO2")': 'B35_SO2', 'S33.Zn.H2S.("H2S")': 'B35_H2S',
    'S8.P.P': 'burner_inputP'
}


class SimpleTabularMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleTabularMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(32, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def zscore_frame(df, cols, mean, std):
    std_safe = std[cols].replace(0, 1)
    return (df[cols] - mean[cols]) / std_safe


def load_lhs_steady_state_sources(path_pattern, required_cols):
    paths = sorted(glob.glob(path_pattern))
    if not paths and os.path.exists(path_pattern):
        paths = [path_pattern]
    if not paths:
        raise FileNotFoundError(f'No LHS steady-state files found: {path_pattern}')

    dfs = []
    for path in paths:
        ext = os.path.splitext(path)[1].lower()
        if ext in ['.xlsx', '.xlsm', '.xls']:
            df = pd.read_excel(path, sheet_name=0, header=2)
            df = df.iloc[1:].dropna(how='all').copy()
        elif ext == '.csv':
            df = pd.read_csv(path)
        else:
            raise ValueError(f'Unsupported LHS source file type: {ext}')

        if 'Status' in df.columns:
            df = df[df['Status'] == 'Run Completed'].copy()

        rename_map = {raw_name: py_name for raw_name, py_name in FULL_MAPPING.items() if raw_name in df.columns}
        df = df.rename(columns=rename_map)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f'Missing required columns in {path}: {missing}')

        dfs.append(df[required_cols].apply(pd.to_numeric, errors='coerce'))

    combined = pd.concat(dfs, ignore_index=True).dropna().reset_index(drop=True)
    if len(combined) == 0:
        raise ValueError('No usable LHS steady-state rows after numeric conversion/dropna.')

    print(f'[Info] Loaded {len(paths)} LHS steady-state file(s), usable rows={len(combined)}')
    for path in paths:
        print(f'  - {path}')
    return combined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--samples', type=int, default=200)
    parser.add_argument('--delta-std', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--lhs-source',
        default='./data/Claus_steady_state/lhs_generated_dynamic_ss_data*.xlsx',
        help='LHS steady-state Excel/CSV path or glob pattern.'
    )
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    rng = np.random.default_rng(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() and config['training'].get('device') == 'cuda' else 'cpu')

    tab_input_cols = [
        'air2_SP',
        'HEATER2_output_T_SP',
        'acidgas_Fm',
        'acidgas_P',
        'acidgas_T',
    ]
    full_model_target_cols = [
        'B35_H2S',
        'B35_SO2',
    ]
    qv_cols = config['training'].get('gain_target_qv', ['B35_H2S', 'B35_SO2'])
    mv_cols = config['training'].get('gain_target_mv', ['air2_SP', 'HEATER2_output_T_SP'])

    tab_mean = pd.read_csv('./results/Tabular_MLP_New/zscore_mean.csv', index_col=0).squeeze('columns')
    tab_std = pd.read_csv('./results/Tabular_MLP_New/zscore_std.csv', index_col=0).squeeze('columns').replace(0, 1)

    df = load_lhs_steady_state_sources(args.lhs_source, tab_input_cols)

    sample_n = min(args.samples, len(df))
    sample_idx = rng.choice(len(df), sample_n, replace=False)
    base_df = df.iloc[sample_idx][tab_input_cols].reset_index(drop=True).astype('float32')

    model = SimpleTabularMLP(input_dim=len(tab_input_cols), output_dim=len(full_model_target_cols)).to(device)
    model.load_state_dict(torch.load('./saved_models/Tabular_MLP_5in_2out_QV.pth', map_location=device))
    model.eval()

    target_mean = torch.tensor(tab_mean[full_model_target_cols].values, dtype=torch.float32, device=device)
    target_std = torch.tensor(tab_std[full_model_target_cols].values, dtype=torch.float32, device=device)

    rows = []
    for mv in mv_cols:
        delta = args.delta_std * float(tab_std[mv])
        plus_df = base_df.copy()
        minus_df = base_df.copy()
        plus_df[mv] += delta
        minus_df[mv] -= delta

        x0_z = torch.tensor(zscore_frame(base_df, tab_input_cols, tab_mean, tab_std).values, dtype=torch.float32, device=device)
        xp_z = torch.tensor(zscore_frame(plus_df, tab_input_cols, tab_mean, tab_std).values, dtype=torch.float32, device=device)
        xm_z = torch.tensor(zscore_frame(minus_df, tab_input_cols, tab_mean, tab_std).values, dtype=torch.float32, device=device)
        x0_z.requires_grad_(True)

        y0_z = model(x0_z)
        y0_p = y0_z * target_std + target_mean

        with torch.no_grad():
            yp_p = model(xp_z) * target_std + target_mean
            ym_p = model(xm_z) * target_std + target_mean

        for qv in qv_cols:
            out_idx = full_model_target_cols.index(qv)
            grad_outputs = torch.zeros_like(y0_p)
            grad_outputs[:, out_idx] = 1.0
            grads_x, = torch.autograd.grad(
                outputs=y0_p,
                inputs=x0_z,
                grad_outputs=grad_outputs,
                retain_graph=True
            )
            mv_idx = tab_input_cols.index(mv)
            autograd_gain = (grads_x[:, mv_idx] / float(tab_std[mv])).detach().cpu().numpy()
            fd_gain = ((yp_p[:, out_idx] - ym_p[:, out_idx]) / (2.0 * delta)).detach().cpu().numpy()
            y0 = y0_p[:, out_idx].detach().cpu().numpy()
            yp = yp_p[:, out_idx].detach().cpu().numpy()
            ym = ym_p[:, out_idx].detach().cpu().numpy()

            for i in range(sample_n):
                rows.append({
                    'sample_idx': int(sample_idx[i]),
                    'mv': mv,
                    'qv': qv,
                    'delta': delta,
                    'base_mv': float(base_df.loc[i, mv]),
                    'y_minus': float(ym[i]),
                    'y_base': float(y0[i]),
                    'y_plus': float(yp[i]),
                    'fd_gain': float(fd_gain[i]),
                    'autograd_gain': float(autograd_gain[i]),
                    'fd_sign': int(np.sign(fd_gain[i])),
                    'autograd_sign': int(np.sign(autograd_gain[i])),
                })

    out_dir = './results/PGIN_Visualizations'
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, 'ann_gain_pair_diagnostics.csv')
    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_csv, index=False)

    print(f'[Done] Saved per-sample diagnostics: {out_csv}')
    for (qv, mv), g in result_df.groupby(['qv', 'mv']):
        fd_pos = (g['fd_gain'] > 0).mean() * 100
        ag_pos = (g['autograd_gain'] > 0).mean() * 100
        sign_match = (g['fd_sign'] == g['autograd_sign']).mean() * 100
        print(
            f'{qv} | {mv}: '
            f'FD mean={g["fd_gain"].mean():.6g}, FD pos={fd_pos:.1f}%, '
            f'Autograd mean={g["autograd_gain"].mean():.6g}, Autograd pos={ag_pos:.1f}%, '
            f'FD/Autograd sign match={sign_match:.1f}%'
        )


if __name__ == '__main__':
    main()
