# 修改歷程

最新的在最上面。版本編號 #1–#10 對應 `VERSIONS.md`。2026-11-12 以前的完整歷程（含開放世界）在原 repo
[woody0115/HSF_fusion](https://github.com/woody0115/HSF_fusion) 的 `CHANGELOG.md`。

| 日期 | 步驟 | 類型 | 內容 | 結果 |
|---|---|---|---|---|
| 2026-11-12 | – | 整理 | 從 HSF_fusion（commit `e54ec54`）拆出封閉集融合，建立本 repo（全新 git 歷史）。移除開放世界部分：`hsf/owod/`、`train_owod.py` 與 OWOD 診斷工具、`owod_tasks.yaml`、文字嵌入、`UniOW_orig/`；`engine.py` 拿掉 event-ignore loss，`dataset.py` 拿掉 ovd／owod 切分，`splits.yaml` 只留驗證集，demo、release 打包、排程腳本、VS Code 設定拿掉 owod 選項。封閉集的實驗紀錄（runs/ 的 step3–6、step8、封存與測試）全部搬過來 | 程式可正常匯入、HSF 模型可建立（23.9 M 參數）；原 repo 不受影響 |
