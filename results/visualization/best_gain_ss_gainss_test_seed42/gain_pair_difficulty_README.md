# 四模型 test：增益配對難度分析

模型：No-Gain、Gain=250、SS-only=2.5、Gain+SS=(2.5,2.5)，seed42。使用剛才四模型 test KCI 的原始逐點明細，不重新訓練或推論。

## 圖與數據

- [teacher_gain_landscape PNG](teacher_gain_landscape.png)／[PDF](teacher_gain_landscape.pdf)／[灰階](teacher_gain_landscape_grayscale.png)
- [conditional_kci_by_gain_magnitude PNG](conditional_kci_by_gain_magnitude.png)／[PDF](conditional_kci_by_gain_magnitude.pdf)／[灰階](conditional_kci_by_gain_magnitude_grayscale.png)
- [conditional_kci_by_teacher_sign PNG](conditional_kci_by_teacher_sign.png)／[PDF](conditional_kci_by_teacher_sign.pdf)／[灰階](conditional_kci_by_teacher_sign_grayscale.png)

- [Teacher 統計](teacher_gain_summary.csv)／[逐點 teacher 增益](teacher_gain_landscape.csv)
- [Q1～Q4 邊界](gain_magnitude_quartile_edges.csv)／[依增益大小的 KCI](conditional_kci_by_gain_magnitude.csv)
- [依正負方向的 KCI](conditional_kci_by_teacher_sign.csv)／[每個操作點的36次比較](gain_pair_probe_accuracy.csv)
- [來源與計算定義](gain_pair_difficulty_protocol.json)／[核對紀錄](gain_pair_difficulty_validation.json)

## 定義

Teacher 增益取同一個 probe 的 plus 方向、第1個 post-step 時刻作為唯一參考，避免重複計數；模型的KCI仍使用±方向各18個時刻，共36次比較。

正規化增益為 teacher_gain × teacher輸入scale / teacher輸出scale。使用保存的Tabular-MLP尺度；這份程式原先圖註的IQR已修正為保存尺度，數值公式維持原樣。

Q1～Q4在每個區域、每個增益配對的有效teacher探測點內各自計算25/50/75百分位；四個模型共用No-Gain teacher邊界。Q1為較小增益，Q4為較大增益；原本無效／近零增益不納入條件式KCI。

誤差棒為2,000次bootstrap的95%區間。重抽樣單位是操作點，其36次方向／時刻比較保留為一整組；不是把36次比較當成獨立樣本，也不是跨訓練seed的不確定性。

沒有有效探測點的正／負增益組別保留NaN，不畫成0%。所有四分位與正負組別相加後，correct與valid均重現原本test KCI計數。

穩態test核心4385點、近外插2692點；KCI是ANN-reference方向一致性。Teacher大小與正負分布反映參考模型與本批探測點，不等同實際製程的物理增益真值。

## 重跑與重畫

完整分析：`python -m scripts.analysis.analyze_gain_pair_difficulty`。
只用保存CSV重畫：`python results/visualization/best_gain_ss_gainss_test_seed42/rebuild_gain_pair_difficulty.py`。
