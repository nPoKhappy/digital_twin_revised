# 實驗設定與輸出索引

2026-11-04 分類完成時的狀態；完整機器可讀對照見 [experiment_catalog.csv](../results/experiment_catalog.csv)。
每個 YAML 的 `output.results_dir` 對應其 config 分類，程式再附加 `exp_name`。Checkpoint 仍在 `saved_models/`。

| 分類 | Gain | SS | 設定 | 結果位置 | 權重／統計 |
| --- | --- | --- | --- | --- | --- |
| `pgin/difficult_gradnorm` | 0.05 | 0 | [difficult_gradnorm_ratio0p005_no_ss_seed42](../configs/pgin/difficult_gradnorm/difficult_gradnorm_ratio0p005_no_ss_seed42.yaml) | [pgin_gain005_ratio0p005_seed42_difficult_group_gradnorm_no_ss](../results/pgin/difficult_gradnorm/pgin_gain005_ratio0p005_seed42_difficult_group_gradnorm_no_ss/) | checkpoint=有；統計=有 |
| `pgin/difficult_gradnorm` | 0.05 | 0 | [difficult_gradnorm_ratio0p01_no_ss_seed42](../configs/pgin/difficult_gradnorm/difficult_gradnorm_ratio0p01_no_ss_seed42.yaml) | [pgin_gain005_ratio0p01_seed42_difficult_group_gradnorm_no_ss](../results/pgin/difficult_gradnorm/pgin_gain005_ratio0p01_seed42_difficult_group_gradnorm_no_ss/) | checkpoint=有；統計=有 |
| `pgin/difficult_gradnorm` | 0.05 | 0 | [difficult_gradnorm_ratio0p02_no_ss_seed42](../configs/pgin/difficult_gradnorm/difficult_gradnorm_ratio0p02_no_ss_seed42.yaml) | [pgin_gain005_ratio0p02_seed42_difficult_group_gradnorm_no_ss](../results/pgin/difficult_gradnorm/pgin_gain005_ratio0p02_seed42_difficult_group_gradnorm_no_ss/) | checkpoint=有；統計=有 |
| `pgin/difficult_gradnorm` | 0.05 | 0 | [gain005_seed42_difficult_group_gradnorm_no_ss](../configs/pgin/difficult_gradnorm/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42_difficult_group_gradnorm_no_ss.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42_difficult_group_gradnorm_no_ss](../results/pgin/difficult_gradnorm/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42_difficult_group_gradnorm_no_ss/) | checkpoint=有；統計=有 |
| `pgin/gain` | 0.05 | 0 | [gain005_seed42](../configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42](../results/pgin/gain/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_seed42/) | checkpoint=有；統計=有 |
| `pgin/gain` | 2500.0 | 0 | [gain2500_seed42](../configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain2500_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain2500_seed42](../results/pgin/gain/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain2500_seed42/) | checkpoint=有；統計=有 |
| `pgin/gain` | 250.0 | 0 | [gain250_seed42](../configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain250_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain250_seed42](../results/pgin/gain/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain250_seed42/) | checkpoint=有；統計=有 |
| `pgin/gain` | 25.0 | 0 | [gain25_seed42](../configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain25_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain25_seed42](../results/pgin/gain/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain25_seed42/) | checkpoint=有；統計=有 |
| `pgin/gain` | 2.5 | 0 | [gain2p5_seed42](../configs/pgin/gain/transformer_layerwise_63var_decoder_input_sp_pgin_gain2p5_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain2p5_seed42](../results/pgin/gain/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain2p5_seed42/) | checkpoint=有；統計=有 |
| `pgin/gain_ss` | 0.05 | 0.05 | [gain005_ss005_seed42](../configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain005_ss005_seed42.yaml) | `results/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain005_ss005_seed42/`（待產生） | 待重跑 |
| `pgin/gain_ss` | 2500 | 2500 | [gain2500_ss2500_seed42](../configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain2500_ss2500_seed42.yaml) | `results/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain2500_ss2500_seed42/`（待產生） | 待重跑 |
| `pgin/gain_ss` | 250 | 250 | [gain250_ss250_seed42](../configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain250_ss250_seed42.yaml) | `results/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain250_ss250_seed42/`（待產生） | 待重跑 |
| `pgin/gain_ss` | 25 | 25 | [gain25_ss25_seed42](../configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain25_ss25_seed42.yaml) | `results/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain25_ss25_seed42/`（待產生） | 待重跑 |
| `pgin/gain_ss` | 2.5 | 2.5 | [gain2p5_ss2p5_seed42](../configs/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_pgin_gain2p5_ss2p5_seed42.yaml) | `results/pgin/gain_ss/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_gain2p5_ss2p5_seed42/`（待產生） | 待重跑 |
| `pgin/no_gain` | 0 | 0 | [no_gain_seed42](../configs/pgin/no_gain/transformer_layerwise_63var_decoder_input_sp_pgin_no_gain_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed42](../results/pgin/no_gain/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_no_gain_seed42/) | checkpoint=有；統計=有 |
| `pgin/ss_only` | 0 | 0.05 | [ss005_only_seed42](../configs/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_pgin_ss005_only_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss005_only_seed42](../results/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss005_only_seed42/) | checkpoint=有；統計=有 |
| `pgin/ss_only` | 0 | 2500.0 | [ss2500_only_seed42](../configs/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_pgin_ss2500_only_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss2500_only_seed42](../results/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss2500_only_seed42/) | checkpoint=有；統計=有 |
| `pgin/ss_only` | 0 | 250.0 | [ss250_only_seed42](../configs/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_pgin_ss250_only_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss250_only_seed42](../results/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss250_only_seed42/) | checkpoint=有；統計=有 |
| `pgin/ss_only` | 0 | 25.0 | [ss25_only_seed42](../configs/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_pgin_ss25_only_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss25_only_seed42](../results/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss25_only_seed42/) | checkpoint=有；統計=有 |
| `pgin/ss_only` | 0 | 2.5 | [ss2p5_only_seed42](../configs/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_pgin_ss2p5_only_seed42.yaml) | [transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss2p5_only_seed42](../results/pgin/ss_only/transformer_layerwise_63var_decoder_input_sp_PGIN_From_Scratch_ss2p5_only_seed42/) | checkpoint=有；統計=有 |
| `steady_state` |  |  | [tabular_mlp_claus](../configs/steady_state/tabular_mlp_claus.yaml) | `results/steady_state/Tabular_MLP_Claus_Final/`（待產生） | checkpoint=有；統計=無 |

## 穩態 ANN teacher

既有 teacher 是 `saved_models/Tabular_MLP_5in_2out_QV.pth`，統計量與原有 ANN 產物在 [results/steady_state/Tabular_MLP_New](../results/steady_state/Tabular_MLP_New/)。`train_tabular_v2` 及 `predict_tabular_v2` 使用內建設定；表格中的 `Tabular_MLP_Claus_Final` 是另一份歷史 YAML，並未宣稱已產生對應模型。

## Gain+SS

五組 `(Gain, SS)` 為 `(0.05, 0.05)`、`(2.5, 2.5)`、`(25, 25)`、`(250, 250)`、`(2500, 2500)`；SS 目標仍為全部 48 個輸出。舊專用 checkpoint、訓練結果及 KCI 批次已刪除，待依 [重跑指令](../configs/pgin/gain_ss/README.md) 重新訓練。

## 評估與圖表

- [KCI test／validation 批次索引](../results/evaluation/gain_eval_all18/README.md)
- [Results 分類總覽](../results/README.md)
- [工作流程與入口](workflow.md)

## 歷史保存

Seed43／44 設定、權重與結果先前已複製至本機 Desktop `seed43_44` 封存並完成 SHA-256 核對；封存不在 Git 中。
混合模型歷史比較保留原數值；其中的舊 Gain+SS 不能作為新五組同步權重的結果。
