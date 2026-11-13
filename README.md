# HSF_Event_RGB_Fusion — Event + RGB 融合物件偵測（自駕車）

重現 *Hierarchical Spatiotemporal Fusion for Event-Visible Object Detection*（Jhong et al., ICRA 2025）。
論文沒有公開程式碼，這裡依論文描述自行實作。本 repo 只含**封閉集融合偵測**。

- **資料集**：DSEC-Detection。標註共 8 類（pedestrian, rider, car, bus, truck, bicycle, motorcycle, train），train 類沒有任何框，實際評估 7 類。
- **基礎模型**：雙流 YOLOv8s。使用 Ultralytics 8.3.158 的元件、loss 與評估函式，不修改 Ultralytics 原始碼。
- **論文目標**：完整 HSF 達到 mAP50 **67.2** ／ mAP50-95 **45.6**。
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

## 下一步

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
