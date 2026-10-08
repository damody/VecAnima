# VecAnima

**正式程式為 Rust CLI**，使用 [twistedfall/opencv-rust](https://github.com/twistedfall/opencv-rust) 綁定本地編譯的 OpenCV 4.14.0。已提供原生 `vectorize` 指令；建置、SDK 與使用方式見 [docs/RUST.md](docs/RUST.md)。Python 保留作為參考／驗證工具。

以《筆觸與色塊分離之動畫向量化》為研究方向的 2D 動畫向量化實驗。詳細規劃見 [PLAN.md](PLAN.md)，目前實作與量測見 [docs/PROGRESS.md](docs/PROGRESS.md)。

目前 Rust 已接入**平滑中心曲線、節點左右線寬／顏色、可變寬彩色筆觸、共享色塊曲線與線性光網格**，另有跨幀追蹤、階段快取與續跑。幾何核心自行實作，不使用 CGAL。這是研究實作，完整長片品質與成本驗收仍待完成；最新狀態見 [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md)。

## Rust CLI

本工作區已備妥 SDK、FFmpeg 與 Rust vendor 依賴，可離線執行：

```powershell
./scripts/cargo.ps1 run --release --locked --offline -- vectorize Isekai.mkv --start 1 --duration 5 --width 640 --colors 32 --out output/my-rust-run
```

輸出包含純向量 SVG、幾何 JSON、原始 PTS 時間軸、resvg 重建 PNG、有聲 MP4、品質報告與 `index.html` 檢視器。輸出目錄須為空；加 `--no-preview` 可略過 MP4 編碼，仍會渲染 SVG 並量測品質。

最新筆觸／填色分離結果見 `output/general-quality-svg/`，另有 `general-quality-svg-scene3`。全時長驗證使用 `output/acceptance-general-320`，完成狀態依其 status.json 及 docs/ACCEPTANCE.md。`verified-clean-layers`、`validated-layers`、`clean-fill-*` 與更早目錄是歷史實驗。新版須重新處理，舊 SVG 不會自動改變。

```powershell
./scripts/cargo.ps1 run --release --locked --offline -- vectorize Isekai.mkv --start 1 --duration 1 --width 640 --fill-model mesh --mesh-vertices 48000 --temporal --out output/new-smooth-run
./scripts/cargo.ps1 run --release --locked --offline -- serve output/new-smooth-run --port 51284
```

瀏覽 `http://127.0.0.1:51284/`；可獨立勾選筆觸／色塊、切換原生／SVG、放大及檢查節點線寬與顏色。單張使用 `image`，有逐幀時長的 JSON 序列使用 `sequence`；同來源、設定與實作的失敗輸出可用 `--resume` 續跑。

## Python 驗證環境（Windows / Python 3.12）

在 PowerShell 的專案目錄執行，需先有 Python 3.12 與 uv：

```powershell
./scripts/setup.ps1
```

setup 使用 `uv.lock` 安裝專案 `.venv`，並依 `scripts/ffmpeg-lock.json` 取得固定版本與 SHA256 的 FFmpeg。工具放在 `.tools`，不更改系統 PATH。已安裝時會校驗兩個執行檔；下載存檔保留供重新安裝使用。

第三方 wheel、授權與 SHA256 清單已存放於 `third_party/`，操作方式見 [third_party/README.md](third_party/README.md)。下載存檔齊全時可執行 `./scripts/setup.ps1 -Offline`，使用本地 `uv.exe` 重建套件與 editable 專案安裝。

## Python 參考與驗證

```powershell
.venv/Scripts/python -m vecanima doctor
.venv/Scripts/python -m vecanima probe Isekai.mkv
.venv/Scripts/python -m vecanima analyze Isekai.mkv --out output/analysis
.venv/Scripts/python -m vecanima vectorize Isekai.mkv --start 1 --duration 5 --width 640 --colors 32 --out output/character
.venv/Scripts/python -m pytest -q
```

輸出目錄必須不存在或為空，不覆寫已有實驗。`--epsilon` 控制輪廓簡化（預設 0.65 像素）；`--stroke-threshold` 控制基準暗線亮度閾值（預設 65）；`--no-preview` 跳過 MP4 與檢視器。這些是實驗參數，不是論文參數的直接對應。

實驗完成後開啟輸出的 `index.html`，或啟動本地伺服器：

```powershell
.venv/Scripts/python -m http.server 51283 --bind 127.0.0.1 --directory output
```

以瀏覽器開啟 `http://127.0.0.1:51283/character/index.html`。伺服器只監聽本機；若該埠已使用，改成其他埠。按 Ctrl+C 停止。

## 輸出說明

| 檔案／目錄 | 內容 |
| --- | --- |
| `source.json` | FFprobe 原始影音資訊 |
| `timeline.json` | 實際來源時間戳、影格持續時間與資產參照 |
| `project.json` | 格式版本、色盤、設定、向量資產索引及限制 |
| `geometry/*.json` | 版本 3 幾何、共享曲線、節點中心／左右線寬／顏色、網格；座標為工作像素 |
| `frames/*.svg` | 純向量影格，沒有內嵌點陣圖片 |
| `original/*.png` | FFmpeg 解碼並縮放的來源影格 |
| `raster/*.png` | 原生線性光網格／SVG 色塊及 Bézier 筆觸的實際重建 |
| `preview.mp4` | 向量結果的光柵預覽，回接來源第一條音訊（若存在） |
| `preview-probe.json` | 預覽影片實際規格 |
| `comparison.jpg` | 左原始影格、右向量重建的四組抽樣對照 |
| `index.html` | 原生 WebGL／SVG、圖層勾選、逐幀／縮放／物件檢視 |
| `report.json` | MAE、PSNR、幾何量、大小與耗時 |
| `status.json` | incomplete / complete 狀態，失敗原因與工具版本 |

色盤由區間內最多 12 張取樣影格共同建立，所有幀共用。只有解碼後**完全相同**的影格會重用幾何；近似影格不凍結。音訊從實際第一張輸出影格的來源時間開始，切片不包含影格前不足一幀的時間。

正式 profile 模型以平滑曲線重建；`--stroke-model baseline` 保留像素遮罩基準供比較。網格及 SVG 仍有離散／顏色近似誤差，窄區域需限制平滑以保持拓撲。短粗陰影以 `--stroke-min-aspect` 留在填色層，幾何辨識不等同完美語意辨識；SVG 體積與跨幀穩定仍需長片驗收。

本地 Isekai 素材與所有輸出均忽略於 Git。影片的字幕串流不燒錄，HDR 色調映射與多音軌選擇尚未實作。

## 乾淨填色與本地 AI 邊界

最新細節輸出為 `output/general-quality-svg`，另有 `general-quality-svg-scene2`（20 秒）及 `general-quality-svg-scene3`（40 秒）；全時長輸出為 `output/acceptance-general-320`，驗收狀態以 [docs/ACCEPTANCE.md](docs/ACCEPTANCE.md) 為準。筆觸移除採完整淡邊支撐、來源暗谷證據及乾淨色彩核心重建，避免殘邊變成陰影，也避免 AI 邊界誤刪真實漸層。加 `--line-model third_party/anilines/basic.onnx` 使用本地 AniLines 邊界；正式推論由 Rust/OpenCV DNN 執行。顏色取自原圖，線寬優先量測來源剖面；AI 約束／推估段標記 inferred 與較低 confidence。模型安裝及來源見 [third_party/anilines/LOCAL-SETUP.md](third_party/anilines/LOCAL-SETUP.md)。

```powershell
./scripts/cargo.ps1 run --release --locked --offline -- serve output/general-quality-svg --port 51284
```

本輪 43 Rust／28 Python／嚴格 Clippy 通過，含淡邊／不明陰影／真實窄陰影／AI 推論、連續弱線、相鄰陰影線寬及模型快取回歸。進一步採強弱支撐連通、獨立來源墨谷顏色、線性光調和恢復、來源跳變硬界面與連續 SVG 漸層，預設 SVG 色差界限為 2。完整度量及限制見 docs/ACCEPTANCE.md、docs/IMPLEMENTATION.md，錯誤見 docs/ERRORS.md E037–E050。
