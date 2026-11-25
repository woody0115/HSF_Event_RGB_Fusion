# 實驗總覽（封閉集 event + RGB 融合，DSEC-Det）

更新：2026-11-25。每個實驗一個 branch，名稱是 `exp/<編號>-<名稱>`，指向該實驗做完時的 commit。
branch 是**累積**的：`exp/01` → `exp/10` 是同一條線，後面的 branch 包含前面所有實驗的程式與紀錄；
`main` 是從 [woody0115/HSF_fusion](https://github.com/woody0115/HSF_fusion) 拆出的起點。

- **評估**：mAP50-95（%），評估器同 Ultralytics（conf 0.001、IoU 0.7）。
- **決定只看驗證集**（zurich_city_05 + 10，整趟行車），test 只回報；`exp/01` 以前的版本 #1–#10 是在 test 上挑權重（見 `VERSIONS.md`）。
- **短排程**：5 epoch、lr 1e-3；兩條骨幹從乾淨重訓的單流模型初始化（`exp/01` 起）。
- 每一步的完整紀錄在 `CHANGELOG.md`，推導在 `docs/fusion_model.md`、`docs/tuning_log.md`。

## 一覽

| # | branch | 日期 | 做了什麼 | 主要結果 |
|---|---|---|---|---|
| 01 | `exp/01-paper-schedule` | 11-12 – 11-14 | 照論文設定（40 epoch、lr 0.005、batch 8）跑 add ／ TFS ／ AF ／ HSF；發現驗證集被污染，重訓單流並修正評估 | 論文的 TFS ／ HSF 提升**沒有重現**：test add 52.3、TFS 50.4、AF 52.4、HSF 48.7；驗證分數都在第 1–2 epoch 最高後下降。乾淨單流：RGB 驗證 42.7 ／ test 50.1，event 21.2 ／ 29.9 |
| 02 | `exp/02-reliability-fusion` | 11-14 | 依可靠度加權的融合三輪：v1 凸組合、v2 各自的 Wiener 增益（`relp2_add`）、v3 分支輔助 loss | v1 退化成 RGB 單流；v2 `relp2_add` 驗證 43.1（add 42.7，單一 seed）；v3 失敗（38.6 ／ 39.6） |
| 03 | `exp/03-softmax-fusion` | 11-14 | Softmax 模態權重（`sw_add` ／ `sw2_add`） | 驗證 42.2 ／ 42.6，沒贏 add；test 最高（51.9）；event 權重在所有光線下都固定約 0.2 |
| 04 | `exp/04-corruption-eval` | 11-15 | RGB 破壞穩健性評估：9 種破壞 × 5 個強度，event 保持乾淨 | mPC：RGB 單流 17.0、add 22.5、`relp2_add` 22.8、`sw_add` 19.1；RGB 變差時融合多約 5.5 分 |
| 05 | `exp/05-cafr-fusion` | 11-15 | FRN 的 CAFR 融合（`cafr` ／ `cafrx`） | `cafr` mPC 23.1（最高），但學到的殘差係數約 0，實際接近 add；`cafrx` 驗證只有 38.8 |
| 06 | `exp/06-diagnose-conditions` | 11-15 – 11-16 | 為什麼不同序列結果差很多；README「各情況最佳的模型」；交接文件 | 大幅差異多是少數類別造成的假象；顯著差異只有 1–3 分；**真正弱點是夜間行人**（RGB 33.7，融合 15.7–28.4） |
| 07 | `exp/07-multi-seed` | 11-16 – 11-17 | add ／ `relp2_add` ／ `sw2_add` ／ `cafr` 各 3 個 seed | 驗證集分不出來（42.6–43.0，差距小於 seed 標準差）；唯一三個 seed 一致的是 RGB 破壞：`cafr`、`relp2_add` > add > `sw2_add` |
| 08 | `exp/08-tuning-sw2add` | 11-17 – 11-18 | 5 輪調參，目標是 RGB 變差時的穩健性與夜間行人；最後採用 C：凍結骨幹、15% 破壞 ＋ 5% RGB 歸零、微調 2 epoch | C 對 `sw2_add`：驗證 val-mPC +13.9、RGB 全黑時 0 → 23.4，乾淨只掉 0.34；test mPC 20.2 → 36.3。event 權重開始隨畫面改變 |
| 09 | `exp/09-literature-experiments` | 11-19 – 11-23 | 依文獻的單一變因實驗（一致性損失、組合破壞、夜間蒸餾等）與組合；夜間錯誤分解；不需訓練的推論改良 | **G3（C ＋ 所有影像蒸餾）最好**：夜間行人 +2.47 ± 0.34、all +0.40（3 個 seed 同向）。夜間行人主要是分數低，不是漏掉。放大 TTA（832 px）all +2.9、夜間行人 +4.4，進步來自解析度（RGB 單流放大也 +2.8） |
| 10 | `exp/10-peod-data` | 11-24 – 11-25 | 下載 PEOD（第二個資料集，和 DSEC 分開訓練與回報）與 NightOwls；PEOD 轉成和 DSEC 相同的訓練格式 | NightOwls 完成（130,064 張）。PEOD 轉換程式完成並驗證對齊；下載卡在 Google Drive 配額：rgb 99%，event 約 146 GB 待下載 |

## 目前的結論

- **目前推薦**：`sw2_add` ＋ 微調 C ＋ 所有影像蒸餾（G3，`exp/09`）。依驗證集挑選，乾淨畫面和 RGB 單流同級，RGB 變差或全黑時明顯較好。
- **論文的 TFS ／ HSF 提升沒有重現**；我們的 RGB 基準比論文高約 9 點，所以都和自家基準比較。
- **單一 seed 的差距不可信**：同一個模型換 seed，test 可以差 2 分，大於多數方法之間的差距。
- **夜間行人是最大弱點**，蒸餾和放大輸入有幫助；下一步考慮用較高解析度訓練或加 P2 偵測頭。
- **PEOD** 下載完成後，作為第二個 benchmark 分開訓練（依 2026-11-12 實驗室會議）。

## 怎麼切換到某個實驗

```bash
git fetch origin
git switch exp/08-tuning-sw2add      # 本機會建立追蹤 origin 的 branch
```

訓練權重不在 repo 裡（只收文字紀錄）；初始權重見 `VERSIONS.md` 的 Release。
