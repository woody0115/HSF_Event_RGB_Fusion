# 模型版本總覽（封閉集融合）

開放世界版本（#11、#12）見原 repo [woody0115/HSF_fusion](https://github.com/woody0115/HSF_fusion) 的 VERSIONS.md。

更新：2026-10-06。測試集：DSEC-Det 13 個 test 序列（13,893 張；白天 12,179／偏暗 1,000／夜間 714）。
偵測數字為 mAP50-95（%），評估器同 Ultralytics（conf 0.001、IoU 0.7）。分光線結果來自 `runs/eval_conditions.md`。
各版本權重：GitHub Release `models-2026-10-12`（https://github.com/woody0115/HSF_fusion/releases/tag/models-2026-10-12）。
架構圖、各模組參數量與速度的詳細版：https://claude.ai/code/artifact/fcf8f32a-ac8b-4ec4-bc46-31cc56203e69

## 依用途挑哪個版本

| 用途 | 版本 | 檢查點 |
|---|---|---|
| 整體最準（以白天為主的場景） | RGB 單流 | `runs/step3/rgb_yolov8s/weights/best.pt` |
| 夜間 | add lr1e-3 | `runs/step4/dual_add_lr1e3_e5/best.pt` |
| 偏暗 | AF lr1e-3 | `runs/step5_6/dual_af_lr1e3_e5/best.pt` |
| 融合模型中整體最佳 | AF lr1e-3／iaf_add／iaf_af（都是 53.5） | 見下表 |

## 一、物件偵測（封閉集合，7 類）

| # | 版本 | 步驟 | 設定 | 全部 | 白天 | 偏暗 | 夜間 | 檢查點 |
|---|---|---|---|---|---|---|---|---|
| 1 | RGB 單流 | 3 | YOLOv8s、COCO 預訓練、10 epochs、lr 0.01、batch 16 | **53.9** | **54.1** | 60.6 | 42.2 | `step3/rgb_yolov8s/weights/best.pt` |
| 2 | Event 單流（GEAR） | 3 | 同上 | 32.9 | 32.9 | 39.0 | 26.7 | `step3/event_yolov8s/weights/best.pt` |
| 3 | add e20 | 4 | 20 epochs、lr 0.005；best = epoch 8 ※ | 51.9 | 51.7 | 59.9 | 44.7 | `step4/dual_add_e20/best.pt` |
| 4 | AF e20 | 5 | 20 epochs、lr 0.005；best = epoch 1 ※ | 52.9 | 53.1 | 60.1 | 41.5 | `step5_6/dual_af_e20/best.pt` |
| 5 | HSF e20（TFS+AF） | 6 | 20 epochs、lr 0.005、4 張片段；best = epoch 16 ※ | 51.5 | 51.6 | 59.0 | 44.1 | `step5_6/dual_hsf_e20/best.pt` |
| 6 | add lr1e-3 | 4 | 5 epochs、lr 0.001、warmup 1 | 53.0 | 53.1 | 61.0 | **45.5** | `step4/dual_add_lr1e3_e5/best.pt` |
| 7 | AF lr1e-3 | 5 | 同 6 | 53.5 | 53.8 | **62.8** | 42.5 | `step5_6/dual_af_lr1e3_e5/best.pt` |
| 8 | HSF lr1e-3 | 6 | 同 6 | 53.1 | 53.2 | 61.2 | 42.5 | `step5_6/dual_hsf_lr1e3_e5/best.pt` |
| 9 | iaf_add（光線閘門） | 8 | 同 6 + 閘門 + 15% modality dropout | 53.5 | 53.7 | 60.9 | 42.5 | `step8_iaf/dual_iaf_add_lr1e3_e5/best.pt` |
| 10 | iaf_af（光線閘門） | 8 | 同 9 | 53.5 | 53.5 | 62.1 | 42.9 | `step8_iaf/dual_iaf_af_lr1e3_e5/best.pt` |

※ 3–5 使用舊的 event 輸入格式（未做 BGR→RGB、[-1,1] 解碼），event 分支等於沒用到預訓練；6 以後是修正後格式。
所有雙流模型的兩個骨幹都從 1、2 初始化。

參考值（不需訓練，`tools/eval_conditions.py` 的 `route_*`）：

| 版本 | 說明 | 全部 | 白天 | 偏暗 | 夜間 |
|---|---|---|---|---|---|
| route_add | 白天用 1，偏暗／夜間用 6 | 53.8 | 54.1 | 60.8 | 45.5 |
| route_af | 白天用 1，偏暗／夜間用 7 | 53.8 | 54.1 | 62.9 | 42.5 |

論文數字（HSF，ICRA 2025）：RGB 44.4、event 24.8、add 43.3、AF 44.5、HSF 45.6。

## 二、封存、測試用（不要拿來用）

| 資料夾 | 原因 | 大小 |
|---|---|---|
| `runs/_invalid_frozen_backbones/` | 載入 Ultralytics 權重後 requires_grad 沒打開，骨幹被凍結 | 約 0.85 GB |
| `runs/_invalid_event_input_format/` | event 輸入格式修正前跑到一半的 add lr1e-3 | 125 MB |
| `runs/step3/event_yolov8s_pipetest/` | event 單流的流程測試 | 49 MB |
| `runs/step4/dual_add_smoke/`、`runs/step5_6/dual_*_smoke/` | 雙流流程測試（2 個短 epoch） | 約 0.47 GB |

大小是原始 run 資料夾含權重在本機的大小；本 repo 只收文字紀錄（log、results.csv、設定）。這些紀錄**一律保留不刪**，作為專案歷程的一部分（見 `CHANGELOG.md`）。

## 三、程式版本對應

| 功能 | 開關 | 程式 |
|---|---|---|
| 融合方式 | `train_dual.py --fusion add / tfs / af / hsf / iaf_add / iaf_af` | `hsf/models/fusion.py` |
| 光線閘門的 modality dropout | `--modality-drop 0.15` | `hsf/models/dual_yolov8.py` |
| 分光線評估 | `eval_conditions.py --models ...` | `tools/eval_conditions.py` |
