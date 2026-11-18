# 交接（2026-11-16）

新對話先讀這份，再看 `README.md`（結果表、「各情況最佳的模型」）、`CHANGELOG.md`（最新在上）與
`docs/fusion_model.md` 第 10–14 節。

## 1. 現況

- 方向：封閉集 Event + RGB 融合（DSEC-Det，雙流 YOLOv8s，P3／P4／P5 融合，GEAR event 輸入）。開放世界（OWOD）暫停，程式留在原 repo
  `HSF_fusion`，不刪。
- 評估規則：在驗證集（zurich_city_05 + 10）挑權重與做決定，test 只回報，不拿 test 調參。
- 短排程：5 epoch、lr 1e-3、warmup 1、seed 0、乾淨初始權重（`runs/rel_iter.sh`）。

## 2. 分支（都只在本機，沒有 push）

依序疊在一起，每條是一個方案：

| 分支 | 內容 | 文件 |
|---|---|---|
| `reliability-fusion` | 可靠度融合三輪（`rel_` ／ `relp_` ／ `rel2_` ／ `relp2_`、輔助 loss） | 第 10 節 |
| `softmax-fusion` | Softmax 模態權重（`sw_` ／ `sw2_`，使用者提議） | 第 11 節 |
| `corruption-eval` | 方案 A：RGB 破壞穩健性（`hsf/data/corruptions.py`、`tools/eval_corruption.py`） | 第 12 節 |
| `cafr-fusion` | 方案 B：FRN 的 CAFR（`cafr` ／ `cafrx`） | 第 13 節 |
| `diagnose-conditions` | 每序列診斷（`tools/diagnose_conditions.py`）、各情況最佳模型表 | 第 14 節 |
| `multi-seed`（目前，2026-11-16） | 4 個主要融合各 3 個 seed（`runs/seed_runs.sh`、`tools/summarize_seeds.py`） | 第 15 節 |

commit 作者：`git -c user.name=walkerp0 -c user.email="$(git log -1 --format=%ae 520dbaf)" commit`，訊息結尾
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`。

## 3. 主要結論

- 乾淨畫面融合只比 RGB 單流多約 1 分；RGB 被破壞時多約 5.5 分（mPC：RGB 17.0、add 22.5、`relp2_add` 22.8、`cafr` 23.1）。
- `sw_add` ／ `sw2_add` 的 event 權重固定約 0.2，乾淨時 test 最高（51.9），破壞時融合裡最差。
- `cafr` 學到的殘差係數接近 0，實際上接近 add；`cafrx` 5 epoch 不夠（驗證集 38.8）。
- 每序列的大幅差異主要是少數類別造成的假象（interlaken_00_a 只有 2 個 bus 框）；顯著差異只有 1–3 分；
  真正的弱點是夜間行人（zurich_city_12_a：RGB 33.7，融合 15.7–28.4）。
- 調參 5 輪（2026-11-17，`docs/tuning_log.md`）：`sw2_add` 凍結兩條 backbone，短微調 2 epoch（15% RGB 破壞 ＋ 5% RGB 歸零），
  驗證集 val-mPC 15.4 → 29.3、RGB = 0 時 0 → 23.4，乾淨只差 −0.34（波動內）。閘門學會看情況切換到 event。
  RGB 歸零時融合模型原本只剩 0–6 分，因為 neck 和 head 沒學過只靠 event 偵測。夜間行人沒有明確改善。
  test 回報（2026-11-18）方向一致：RGB 破壞 mPC 20.2 → 36.3，乾淨 −0.46，RGB = 0 時 29.8。test 上 RGB 單流的夜間行人 40.6，
  仍高於所有融合模型（34.9–36.6）。
- 單一 seed 下各融合之間的差距大多在雜訊內。
- 3 個 seed（2026-11-16）確認：驗證集上 add、`relp2_add`、`sw2_add`、`cafr` 分不出來（42.6–43.0，標準差 0.2–0.5，
  和 RGB 單流 42.7 也差不多）；三個 seed 方向都一致的只有 RGB 破壞 mPC（test）：`cafr` ≈ `relp2_add` 23.1 > add 22.3 > `sw2_add` 20.0。

## 4. 下一步（每一項都要先問使用者同意）

1. ~~add、`relp2_add`、`sw2_add`、`cafr` 各跑 3 個 seed~~（2026-11-16 完成，第 15 節）。新提議：在驗證集上做 RGB 破壞評估，
   讓穩健性也能依規則用來挑模型（只跑評估，不用訓練）。
2. 評估：序列層級只算框數足夠的類別或以框數加權，附 bootstrap 區間（只改評估程式）。
3. FRN 式 RGB 歸零／RGB 破壞訓練增強（處理 Softmax 權重不動與夜間行人）。
4. 之前提過、仍待同意：輔助 loss 改成每條分支獨立的輕量偵測頭、P3 的負 event 先驗、用乾淨初始權重跑 40 epoch 論文設定（`INIT=_val`）。
5. 要不要 push 到 GitHub（woody0115/HSF_Event_RGB_Fusion）等使用者決定。

## 5. 實驗室伺服器

- JupyterHub `http://140.118.234.8:8000`，使用者 walkerp0，4× RTX A4000。要先連 VPN（Pulse Secure，使用者自己登入）。
- 程式在 `~/work/HSF_Event_RGB_Fusion`。跑 Python 前要 `source ~/work/hsf_env.sh`（ultralytics 在封存的 `.venv`）。
- 長時間工作用 `setsid nohup bash script.sh > log 2>&1 < /dev/null &`，跑完寫一個 marker 檔；直接在終端機跑的指令會隨連線中斷被砍。
- 伺服器時鐘是 UTC。torch 2.6+ 載入自己的 checkpoint 要 `weights_only=False`。
- 操作伺服器用 Jupyter API（終端機 websocket 與 contents API）。API token 只存在本機暫存的 `.jt`，不能印出、不能 commit。

## 6. 規則

- 同時最多用 2 張 GPU（用 GPU 0 與 3）。
- 不刪舊版本與封存（搬移或封存即可）；修改寫進 `CHANGELOG.md` 與文件歷程。
- `camera_url.txt` 含攝影機密碼，不能複製、分享或上傳；不代輸任何密碼或 token。
- 從外部套件庫在伺服器安裝套件前要先問。
- 說明用繁體中文，程式註解用英文。
