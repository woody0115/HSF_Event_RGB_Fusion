# HSF_Event_RGB_Fusion — Event + RGB 融合物件偵測（自駕車）

重現 *Hierarchical Spatiotemporal Fusion for Event-Visible Object Detection*（Jhong et al., ICRA 2025）。
論文沒有公開程式碼，這裡依論文描述自行實作。本 repo 只含**封閉集融合偵測**。

- **資料集**：DSEC-Detection。標註共 8 類（pedestrian, rider, car, bus, truck, bicycle, motorcycle, train），train 類沒有任何框，實際評估 7 類。
- **基礎模型**：雙流 YOLOv8s。使用 Ultralytics 8.3.158 的元件、loss 與評估函式，不修改 Ultralytics 原始碼。
- **論文目標**：完整 HSF 達到 mAP50 **67.2** ／ mAP50-95 **45.6**。
- **實驗總覽**：10 個實驗各有一個 branch（`exp/01-paper-schedule` … `exp/10-peod-data`），目的、結果與對應關係見 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。
- **來源**：從 [woody0115/HSF_fusion](https://github.com/woody0115/HSF_fusion) 的 commit `e54ec54` 拆出。開放世界（DSEC-OWOD）的程式、結果與診斷都留在原 repo，原 repo 完整保留。

## 目前結果（DSEC-Det test，13 個序列、13,893 張）

mAP50-95（%），評估設定同 Ultralytics validator。詳細版本與分光線結果見 `VERSIONS.md`。

| 模型 | 論文 | 我們：lr 1e-3、5 epoch（test 挑權重） | 我們：論文設定（驗證集挑權重） | 比自家 Add 的提升（論文／論文設定） |
|---|---|---|---|---|
| RGB 單流 | 44.4 | **53.9** | – | – |
| Event 單流（GEAR） | 24.8 | 32.9 | – | – |
| Add | 43.3 | 53.0 | 52.3 | 0 ／ 0 |
| TFS only | 44.2 | – | 50.4 | +0.9 ／ −1.9 |
| AF | 44.5 | 53.5 | 52.4 | +1.2 ／ +0.1 |
| 完整 HSF（TFS + AF） | **45.6** | 53.1 | 48.7 | +2.3 ／ −3.6 |

- 我們的 RGB 基準比論文高約 9 點，原因不明（論文未詳述資料與評估細節），因此以「相對自家基準的提升」比較。
- 目前所有融合模型都低於 RGB 單流，論文中 TFS 帶來的提升沒有重現；加了 TFS 的兩組反而最低。
- 論文設定（40 epoch、lr 0.005、batch 8、seed 0，2026-11-12 在實驗室伺服器上跑）：四組的驗證分數都在第 1–2 個 epoch 最高，之後一路下降（例如 add 64.8 → 55.3），test 用的權重都是第 1–2 個 epoch。分光線結果見 `runs/eval_conditions.md`（`*_paper`）。

### 乾淨評估下的結果（2026-11-14，分支 `reliability-fusion`）

單流初始權重改用 train_core 重訓，在整趟行車的驗證集（zurich_city_05 + 10）挑權重；融合用短排程（5 epoch、
lr 1e-3、seed 0）。決定只看驗證集，test 只回報。mAP50-95（%）：

| 模型 | 驗證集 | test 全部 | 白天 | 偏暗 | 夜間 |
|---|---|---|---|---|---|
| RGB 單流 | 42.7 | 50.1 | 50.2 | 60.0 | 36.9 |
| Event 單流 | 21.2 | 29.9 | 29.7 | 38.1 | 22.9 |
| Add | 42.7 | 51.1 | 51.6 | 59.3 | 38.1 |
| `rel_add` v1（凸組合閘門） | 42.2 | 49.9 | 50.1 | **59.8** | 37.1 |
| `rel2_add` v2（物理先驗 + 學習修正 δ） | 42.2 | 50.7 | 51.0 | 59.6 | 36.4 |
| `relp2_add` v2（只用物理先驗） | **43.1** | 50.6 | 50.9 | 58.9 | 37.7 |
| `relp2_add` + 分支輔助 loss（v3） | 38.6 | 48.1 | 48.5 | 56.0 | 35.5 |
| `relp2_add` + event 輔助 loss（v3） | 39.6 | 47.3 | 47.7 | 54.2 | 36.3 |
| `sw_add`（Softmax 權重，和為 1） | 42.2 | **51.9** | **52.0** | **60.4** | 38.9 |
| `sw2_add`（Softmax 權重，和為 2） | 42.6 | 51.7 | 51.9 | 60.2 | 39.1 |
| `cafr`（FRN 的 CAFR，殘差、初始等於 add） | 42.4 | 51.1 | 51.5 | 59.1 | 39.0 |
| `cafrx`（FRN 的 CAFR，無殘差） | 38.8 | 50.7 | 50.8 | 59.5 | **39.8** |

- 乾淨初始權重下，Add 比 RGB 單流高 1.0（夜間 +1.2），融合開始有幫助。
- 可靠度融合三輪的推導與檢討見 `docs/fusion_model.md` 第 10 節；目前只有單一 seed，驗證集上 0.4 的差距還不能下結論。
- Softmax 模態權重（分支 `softmax-fusion`，`docs/fusion_model.md` 第 11 節）：test 較高但驗證集沒有贏 Add；event 權重在所有光線下都固定約 0.2，沒有學會依場景調整。

### RGB 破壞下的穩健性（2026-11-15，分支 `corruption-eval`）

只破壞 RGB 輸入（9 種破壞 × 5 個強度，event 保持乾淨），test 每 2 張取 1 張，mAP50-95（%）；
mPC = 所有破壞與強度的平均，rPC = mPC ／ 乾淨。`python tools/eval_corruption.py --models ...`，結果在 `runs/eval_corruption.md`。

| 模型 | 乾淨 | mPC | rPC | 雜訊 | 模糊 | 霧 | 亮度／對比 | 低光 |
|---|---|---|---|---|---|---|---|---|
| RGB 單流 | 50.0 | 17.0 | 0.34 | 3.9 | 13.5 | 18.1 | 29.8 | 36.6 |
| Add | 51.3 | 22.5 | 0.44 | 7.0 | 21.1 | **28.3** | 35.0 | **41.5** |
| `relp2_add` | 50.8 | 22.8 | **0.45** | 8.3 | 21.0 | 26.9 | 35.3 | 40.5 |
| `sw_add` | **52.0** | 19.1 | 0.37 | 4.1 | 16.1 | 22.5 | 32.6 | 39.3 |
| `sw2_add` | 51.8 | 19.3 | 0.37 | 4.6 | 16.3 | 23.2 | 32.4 | 39.1 |
| `cafr` | 51.2 | **23.1** | **0.45** | 8.7 | 21.4 | 27.7 | **35.6** | 40.4 |
| `cafrx` | 50.7 | 22.2 | 0.44 | **9.4** | **21.6** | 23.2 | 33.0 | 39.5 |

- RGB 變差時融合確實有用：Add 與 `relp2_add` 比 RGB 單流多約 5.5 分。
- Softmax 權重固定押在 RGB（0.8 ／ 0.2），乾淨時最高、破壞時最差；破壞加重時 event 權重沒有上升。
- CAFR（分支 `cafr-fusion`，`docs/fusion_model.md` 第 13 節）：`cafr` mPC 最高（23.1），但學到的殘差係數很小，實際上接近 Add；`cafrx` 驗證集只有 38.8。

### 為什麼不同序列差很多（診斷，2026-11-15，分支 `diagnose-conditions`）

`python tools/diagnose_conditions.py dump ...` ／ `analyze ...`，結果在 `runs/diag/diagnose.md`，推導與數字見
`docs/fusion_model.md` 第 14 節。

- **大幅差異大多是假象**：每個序列的 mAP 是各類 AP 的平均，只有幾個框的類別和上千個 car 框一樣重。interlaken_00_a
  只有 2 個 bus 框，bus AP 從 13 到 80，序列 mAP 就從 40.5 跳到 73.0；car AP 所有模型都在 85–86。
- **顯著的差異只有 1–3 分**：thun_02_a 融合 +1.2 ～ +2.2，偏暗的 interlaken_01_a 約 −2.4。
- **真正的弱點**：夜間行人（zurich_city_12_a：RGB 33.7，融合 15.7–28.4）。
- 亮度、背景事件、event SNR 等場景訊號都解釋不了差異（|ρ| ≤ 0.3）。拿掉 event 後 add ／ `relp2_add` ／ `cafr` 掉 2–7 分，
  `sw_add` 掉得最少。
- 單一 seed 下，各融合方法之間的差距大多在雜訊範圍內。

### 各情況最佳的模型（2026-11-15，短排程、seed 0）

mAP50-95（%）。括號內是第 2 名；差距不到 0.5 視為平手（單一 seed 下分不出來）。決定模型只看驗證集，其他列只是回報。
資料：`runs/eval_conditions.md`、`runs/eval_corruption.md`、`runs/diag/diagnose.md`。

| 情況 | 最佳 | 第 2 名 | RGB 單流 |
|---|---|---|---|
| 驗證集（挑模型的依據） | `relp2_add` 43.1 | add 42.7（RGB 同分） | 42.7 |
| test 全部 | `sw_add` 51.9 | `sw2_add` 51.7（平手） | 50.1 |
| 白天 | `sw_add` 52.0 | `sw2_add` 51.9（平手） | 50.2 |
| 偏暗 | `sw_add` 60.4 | `sw2_add` 60.2（平手，RGB 60.0 也只差 0.4） | 60.0 |
| 夜間 | `cafrx` 39.8 | `sw2_add` 39.1 | 36.9 |
| RGB 破壞整體（mPC） | `cafr` 23.1 | `relp2_add` 22.8（平手） | 17.0 |
| 雜訊（高斯／散粒／脈衝） | `cafrx` 9.4 | `cafr` 8.7 | 3.9 |
| 模糊（失焦／動態） | `cafrx` 21.6 | `cafr` 21.4（平手） | 13.5 |
| 霧 | add 28.3 | `cafr` 27.7 | 18.1 |
| 亮度／對比 | `cafr` 35.6 | `relp2_add` 35.3（平手） | 29.8 |
| 低光（曝光不足 + 散粒雜訊） | add 41.5 | `relp2_add` ／ `cafr` 40.4 | 36.6 |
| 破壞強度 1–4 | `cafr` | `relp2_add`（強度 3、4 平手） | – |
| 破壞強度 5（最強） | `relp2_add` 13.9 | `cafr` 13.8（平手） | – |
| 序列 thun_02_a（最大，3861 張） | `sw_add` 54.3 | add 54.0（平手） | 52.1 |
| 序列 zurich_city_13_a | `rel2_add` 47.4 | `sw_add` 46.3 | 44.8 |
| 夜間行人（zurich_city_12_a） | RGB 單流 33.7 | `relp2_add` 28.4 | 33.7 |

- 序列層級只列出贏 RGB 達到統計顯著的兩段。interlaken_01_a（偏暗）、thun_01_b、zurich_city_12_a、zurich_city_15_a
  是 RGB 單流最高，但最好的融合只輸 0.4–1.7 分。其他序列的第 1 名由少數類別決定，不可靠（見上一節）。
- 沒有一個模型在所有情況都最好。`sw_add` ／ `sw2_add` 是「RGB 乾淨時最好」，RGB 一壞就是融合裡最差的（mPC 19.1 ／ 19.3）。
  add、`relp2_add`、`cafr` 三者在各種破壞下輪流第 1，彼此差距多半在 1 分內，可以視為同一級。
- 想要乾淨與破壞都不差、又按驗證集挑：`relp2_add`（驗證集第 1、mPC 第 2）或 add。要確定還需要 3 個 seed。
  （3 個 seed 的結果見下一節：驗證集上四者分不出來。）

### 3 個 seed（2026-11-16，分支 `multi-seed`）

add、`relp2_add`、`sw2_add`、`cafr` 各跑 seed 0、1、2（短排程同上；`runs/seed_runs.sh`），平均 ± 標準差，mAP50-95（%）。
`python tools/summarize_seeds.py`，每個 seed 的數字在 `runs/seeds_summary.md`；推導見 `docs/fusion_model.md` 第 15 節。

| 模型 | 驗證集 | test | 白天 | 偏暗 | 夜間 | RGB 破壞 mPC | rPC |
|---|---|---|---|---|---|---|---|
| RGB 單流（1 個 seed） | 42.7 | 50.1 | 50.2 | 60.0 | 36.9 | 17.0 | 0.34 |
| add | 42.8 ± 0.3 | 50.5 ± 1.1 | 50.8 ± 1.1 | 58.8 ± 0.5 | 37.3 ± 1.6 | 22.3 ± 0.4 | 0.44 |
| `relp2_add` | 42.9 ± 0.4 | 50.8 ± 0.4 | 51.1 ± 0.3 | 59.8 ± 1.3 | 37.3 ± 0.6 | **23.1 ± 0.4** | **0.45** |
| `sw2_add` | **43.0 ± 0.5** | **51.4 ± 0.3** | **51.6 ± 0.4** | **60.2 ± 0.4** | **38.1 ± 1.2** | 20.0 ± 0.7 | 0.39 |
| `cafr` | 42.6 ± 0.2 | 51.0 ± 0.2 | 51.4 ± 0.2 | 59.5 ± 0.4 | 37.9 ± 1.0 | **23.1 ± 0.4** | **0.45** |

- **驗證集上四者分不出來**：平均 42.6–43.0，差距都小於 seed 間的標準差（0.2–0.5）。`relp2_add` 單一 seed 時領先 add 0.4，
  同 seed 比較變成 +0.42 ／ −0.06 ／ +0.03，是 seed 波動。
- 同一個模型換 seed，test 可以差 2 分（add：51.1、49.1、51.1），比大多數方法之間的差距還大。
- **唯一三個 seed 方向都一致的差異是 RGB 破壞**（test 上算的，只能回報）：同 seed 比 add，`cafr` +0.6 ／ +0.9 ／ +0.9、
  `relp2_add` +0.2 ／ +0.5 ／ +1.7、`sw2_add` −3.2 ／ −2.0 ／ −1.7。
- `sw2_add` 三個 seed 的 event 權重都約 0.20，不隨光線或破壞改變；乾淨 test 最高、RGB 破壞時最差，第 11 節的結論重現。

## 下一步

0. ~~主要方法各跑 3 個 seed~~（2026-11-16 完成，見上一節：驗證集分不出來）。序列層級只看框數足夠的類別或以框數加權，附 bootstrap 區間；夜間行人另外處理（RGB 歸零／RGB 破壞訓練增強）。
0. 若要用 RGB 破壞下的穩健性挑模型，要先在驗證集上做破壞評估（目前只有 test）。
0. 可靠度融合（分支 `reliability-fusion`）：輔助 loss 改成每條分支獨立的輕量偵測頭再試；之後再用乾淨初始權重跑論文設定四組（`INIT=_val`）。
1. 論文設定（40 epoch、lr 0.005）在我們這邊驗證分數一路下降，下一輪要找出原因：較小的 lr ／ 較少的 epoch、TFS 與 HSF 每個 epoch 都評估（目前每 2 個 epoch 一次，沒評估到第 1 個 epoch）。
2. 一律用 `--split val`：在驗證集挑權重，test 只在最後評估一次。
3. 每組 2–3 個 seed（`train_dual.py --seed`）；結果依光線（白天／偏暗／夜間）分開報告。

照論文設定跑四組（`GPUS` 有幾張就同時用幾張，每張 GPU 依序跑分到的組別；共用的實驗室伺服器最多用 2 張）：

```bash
SEED=0 GPUS="0 3" setsid nohup bash tools/run_paper_schedule.sh > runs/paper_schedule_seed0.log 2>&1 < /dev/null &
```

兩條骨幹從單流模型初始化：`runs/step3/rgb_yolov8s/weights/best.pt`、`runs/step3/event_yolov8s/weights/best.pt`
（權重見 `VERSIONS.md` 的 Release，v01、v02）。資料路徑和 `configs/paths.yaml` 不同的機器，另建
`configs/paths.local.yaml`（不進版控）覆寫，例如 `processed_root: /home/user/data/DSEC_train`。

## 流程

| 步驟 | 內容 | 程式 |
|---|---|---|
| 0 | 下載 DSEC-Det（4 連線續傳），時間戳記另外抽出 | `tools/download_dsec_det.ps1`、`download_dsec_det.sh`、`fetch_timestamps.py` |
| 1 | 環境檢查 | `tools/check_env.py` |
| 2 | 前處理：GEAR event 影像、YOLO 標註 | `tools/preprocess_dsec.py`、`build_yolo_view.py`、`check_processed.py` |
| 3 | 單流基準：只用 RGB ／ 只用 event | `tools/train_single.py --modality rgb|event` |
| 4–6 | 雙流融合：add、AF、TFS、HSF | `tools/train_dual.py --fusion add|af|tfs|hsf` |
| 8 | 光線自適應融合（我們的延伸）：`iaf_add`、`iaf_af`，加上硬切換參考值 `route_*` | `tools/train_dual.py --fusion iaf_add`、`tools/eval_conditions.py` |
| – | 分光線評估（白天／偏暗／夜間） | `tools/lighting_stats.py`、`tools/eval_conditions.py` |

步驟編號沿用原 repo；步驟 7、9、10（開放世界）不在本 repo。

## 程式架構

```
hsf/
├── engine.py            雙流訓練／評估迴圈（含 TFS 的連續影格訓練、續訓、驗證集挑權重）
├── data/
│   ├── dataset.py       RGB + GEAR 成對資料集；split：train ／ train_core ／ val ／ test
│   ├── gear.py          GEAR event 表示法（論文 Sec. III-B，Eq. 1–5）
│   └── event_sim.py     從一般影片模擬 event（demo 用）
└── models/
    ├── dual_yolov8.py   雙流 YOLOv8s：兩條骨幹（layers 0–9），在 P3/P4/P5 融合，共用 Neck + Detect
    └── fusion.py        AddFusion、AFFusion、TFSFusion、HSFFusion，以及 IllumGate（iaf_*）
tools/                   下載、前處理、訓練、評估、demo、排程腳本
configs/                 路徑（paths.yaml）、驗證集切分（splits.yaml）、Ultralytics 資料設定
runs/                    封閉集的訓練紀錄與結果（只有文字檔，權重見 VERSIONS.md 的 Release）
```

## 已確認的設計決定

- **GEAR**：事件時窗是前一張影像到這一張影像（sync=front）；c=8、τ=10；tanh 前以灰階單位（÷255）縮放；h、feat 以中間灰為 0，num 以黑色為 0；h 照論文不額外衰減。
- **標註**：框裁到 640×480 內，裁後寬或高小於 2 px 就丟棄；保留 track_id。
- **切分**：13 個 test 序列，同論文的 47/13 切分。2026-10-13 起再從 47 個訓練序列切出 5 個驗證序列（5,413 張，含 1 個夜間序列，`configs/splits.yaml`）；版本 #1–#10 是用全部 47 個序列訓練、在 test 上挑權重（`--split legacy` 可重現）。
- **雙流訓練排程**：論文是 40 epoch。步驟 4–6 的第一輪用 20 epoch（骨幹已由單流預訓練），其餘照論文：SGD lr 0.005、momentum 0.937、batch 8；Ultralytics 式 nominal batch 64 累積梯度、warmup、EMA、AMP。RTX 5070 上約 84 ms/iter，約 12 分鐘/epoch。
- **評估**：與 Ultralytics validator 相同（conf 0.001、IoU 0.7、multi-label、max 300）；`tools/verify_evaluator.py` 實測兩者 mAP 差距小於 0.3 點。
- **AF 模組**（論文 III-D）：自適應混合池化（λa·avg + λm·max）→ 兩模態 token 串接 + 位置編碼 → 1 層 transformer（λsa1/λsa2、λmlp1/λmlp2 自適應殘差）→ 上採樣加回兩個分支 → 串接 + 1×1 Conv 進 Neck。論文沒寫、我們的選擇：池化到 8×8（同 CFT）、1 層、8 頭、MLP ×4；並在「加回分支」與「進 Neck」三處加上初始為 0 的可學習權重，使 AF 起始時等同相加基準。參數 16.2 M → 21.2 M。
- **TFS 模組**（論文 III-C，Eq. 6–9）：event 分支上的卷積 GRU，更新／重置閘門同時看 event 與 RGB 特徵，1×1 卷積；接在每個尺度的 AF 之前。我們的補充：輸出 x_evt + λ_t·h_t（λ_t 初始為 0）。
  - **訓練**：4 張連續影格為一個片段（同一組資料增強），batch = 4 片段 × 4 張，片段內完整 BPTT，每個片段開頭記憶歸零；沒有標註的影格照常更新記憶但不算 loss。
  - **評估**：每個 test 序列依時間順序串流，記憶不中斷（序列開頭歸零），只在有標註的影格算 mAP。片段長度與 batch 是我們的選擇（論文未寫）。
- **光線自適應融合**（步驟 8，`IllumGate`）：每個尺度用小型 MLP 讀 RGB 的亮度直方圖（16 bins）+ 4×4 平均色 + 兩模態特徵的 GAP，輸出每通道閘門 g = 2·sigmoid(·) ∈ (0, 2)；neck = rgb + g·(融合輸出 − rgb)。最後一層初始為 0，起始時等於原本的 add ／ AF。訓練時 15% 影像強制 g = 0（modality dropout）。
- **修正紀錄（2026-10-05）**
  1. Ultralytics checkpoint 的參數 requires_grad=False，最初載入後沒有恢復，導致步驟 4–6 第一輪實際上凍結了骨幹／Neck／Head（結果在 `runs/_invalid_frozen_backbones/`）。已修正並重跑。
  2. 雙流的 event 輸入格式和步驟 3 event 模型訓練時不同（沒做 BGR→RGB、用 [-1,1] 解碼而不是 /255），event 骨幹的預訓練初始化等於失效。已改成和 Ultralytics 一致。**版本 #3–#5（20 epoch）是在舊格式下訓練的。**

## 螢幕影片 demo

```bat
.venv\Scripts\python.exe tools\demo_screen.py --source screen --model af
```

- **畫面來源**：啟動後用滑鼠框出影片播放的區域；`--source` 也可以給影片檔。
- **模型**：`--model af | add | iaf_add | rgb`。
- **按鍵與滑桿**：q 結束、s 存圖、e 顯示／隱藏模擬 event、r 清空 event 記憶；滑桿是 conf、IoU。
- **沒有 event camera**：event 從畫面模擬（`hsf/data/event_sim.py`）。在 DSEC 上，模擬 event 的 AF 模型是 52.9，真實 event 53.5，只用 RGB 53.9（`tools/event_sim_check.py`），所以 demo 只用來展示運作方式。

## 資料位置與環境

- `D:\DSEC\`：原始資料（約 211 GB）；`D:\DSEC_processed\`：前處理結果。路徑集中在 `configs/paths.yaml`。
- 訓練資料建議放 NVMe SSD：大量小檔隨機讀取在傳統硬碟上太慢。

```bat
py -3.13 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe tools\check_env.py
```

開發環境：RTX 5070 12 GB、Python 3.13、PyTorch 2.7.1、Ultralytics 8.3.158。VS Code：`Ctrl+Shift+D` 選設定後按 F5。
