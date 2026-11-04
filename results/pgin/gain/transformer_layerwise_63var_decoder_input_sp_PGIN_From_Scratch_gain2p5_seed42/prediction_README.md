# Gain=2.5 預測重跑

已使用目前已訓練權重重新預測全部 22 組設定資料，覆寫原輸出，未備份。

- [H1／H18／TS 指標](prediction_H1_H18_TS_summary.csv)
- [TS RMSE 獨立重算](prediction_RMSE_recheck.csv)
- [檢查紀錄](prediction_validation.json)
- [執行紀錄](prediction.log)
- [四模型 test 比較](../visualization/four_model_gain2p5_ss001_dg001_test_seed42/README.md)

48 個輸出、1056 筆 TS RMSE 檢查通過，132 筆 H1／H18／TS 與 grouped 輸出一致。14 張個別濃度圖已擴展縱軸以顯示完整預測範圍，數值與指標不變。
