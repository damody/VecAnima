# VecAnima 實作進度

日期：2026-10-08。完整開發路線見 `PLAN.md`。

## Rust CLI 與本地依賴（最新）

正式程式改為 Rust CLI；下列 Python 向量化成果是研究與比對基準，不能視為 Rust 完成狀態。

- Rust `Cargo.lock` 鎖定 98 個外部 crate（含新增 resvg 0.48.1），完整原始碼與授權 vendor 至 `third_party/rust/`；已透過本地來源離線編譯及測試。
- 使用者提供的 `third_party/opencv-4.14.0.zip` 已校驗 SHA256；不是預編譯 SDK。已用本機 MSVC / CMake 編譯 x64 Release 的 core、imgproc、imgcodecs DLL / LIB 至 `.tools/opencv-sdk/`。未使用 4.13.0，原始 ZIP 保留。
- OpenCV 來源、設定與建置資訊在 `scripts/opencv-lock.json` 及 `.tools/opencv-sdk/vecanima-build.json`；原始碼和上游授權保留在 `.tools/opencv-src/opencv-4.14.0/`，SDK 另有 LICENSE / etc/licenses。
- `opencv` crate 0.101.0 使用此 SDK 與既有 libclang；Rust `doctor` 實際灰階轉換成功，回報 OpenCV 4.14.0。
- Rust `probe` / `extract` 已直接呼叫本地 FFmpeg，不依賴 Python 子程序。Isekai 20–25 秒解碼得到 120 幀、4.980 秒；逐幀 PTS、來源時間、duration 與 PNG SHA256 均與 Python 基準一致。
- Rust `edges` 實際讀寫 PNG、執行 Canny；640×360 的角色影格得到 11,363 個邊緣像素，逐像素與 Python OpenCV 參考相同。結果：`output/rust-opencv4140-edges.png`；驗證紀錄：`output/rust-validation.json`。
- Rust `vectorize` 已完成：共用 Lab 色盤、bilateral、最近色盤分割、暗線下補色、自行像素格邊界追蹤、OpenCV 輪廓簡化、孔洞 SVG、resvg 重建、品質報告、有聲預覽與本地 SVG 播放器。核心處理沒有 Python 子程序。
- 精確重複影格共用 SVG／幾何，雜湊候選仍逐位元組確認原始影像相等；重用不更動幀時間。SVG 使用整數座標，減少文字資料量。
- 修正 FFmpeg concat 對 Windows extended path 的相對路徑解析，以及 OpenCV 對中文路徑的讀取：由 Rust 讀取檔案，再以 OpenCV imdecode 解碼。輸出支援空格、單引號及中文路徑。
- `./scripts/cargo.ps1 test --locked --offline`：7 passed。Python 驗證套件：24 passed，其中 5 項呼叫原生 Rust Release CLI，涵蓋 VFR、非零來源起點、影音對齊、孔洞／細線／黑幀、資產重用、重現性、不覆寫、無 MP4 模式及中文路徑。Release 建置及 `doctor` 通過，主程式為 `target/release/vecanima.exe`，透過 `scripts/cargo.ps1` 設定本地 DLL 路徑。
- Python 驗證工具的 21 個 wheel（121,689,183 bytes）、授權、精確 requirements 與雜湊清單已下載至 `third_party/`。從空快取建立另一個虛擬環境，離線安裝並通過全部 19 個 Python 測試。FFmpeg ZIP 與本地 uv 亦可離線校驗。

## Rust 向量化實驗（本輪）

兩段皆為 640×360、32 色、epsilon 0.65、暗線閾值 65、seed 7，使用 Rust Release 與 OpenCV 4.14.0。預覽由 SVG 經 Rust resvg 重建後編碼；兩段均確認 120 幀、音訊存在、時長與時間軸一致。

| 片段 / 目錄 | 幀數 / SVG 資產 | 平均 MAE | 平均 PSNR | 不重複 SVG bytes | 耗時 |
| --- | --- | --- | --- | --- | --- |
| 1–6 秒 / `output/rust-isekai-character/` | 120 / 120 | 4.8567 | 30.4612 dB | 42,202,599 | 11.87 秒 |
| 20–25 秒 / `output/rust-isekai-20s-v2/` | 120 / 106 | 4.4709 | 32.4712 dB | 30,263,798 | 8.98 秒 |

兩段逐幀 PTS、來源時間與 duration 均與 Python 參考相同。品質接近 Python 基準；SVG 文字較小主要來自整數座標與較精簡格式，不能宣稱完成幾何壓縮。耗時包含抽幀、色盤、幾何、渲染、影音編碼及報告；不是多次效能基準，且 Rust 尚未包含 Python 的近靜態差分診斷，不能直接當成嚴格語言效能比較。

已目視檢查兩段抽樣對照，漸層色階、複雜背景細節損失與輪廓尖角仍存在。瀏覽器驗證 SVG 載入、下一幀顯示 2/120（0.042 秒）及縮放。修正手動逐幀時影片 seeked 事件重設 SVG 的問題。

各目錄包含 `index.html`、`preview.mp4`、`comparison.png`、`report.json`。彙整驗證在 `output/rust-vectorization-validation.json`。`output/rust-isekai-20s/` 是先前 concat 路徑失敗的 incomplete 診斷目錄，保留且不當成成功實驗。

下一批進入 M3 正式筆觸模型，建立中心線、局部剖面與非對稱線寬；接著自行實作 Rust 三角化、漸層與時間穩定。未使用 CGAL。

## Python 參考基準已完成

- M0：Python 專案、uv 鎖定依賴、本地 FFmpeg / ffprobe、固定下載版本與 SHA256、setup、doctor。
- M0：Isekai 全片 metadata 與 12 張帶時間縮圖；合成測試目前由 pytest 產生，完整可重用資料集尚未建立。
- M1：準確 seek、PNG 無損抽幀、showinfo 實際 PTS、VFR 持續時間、來源時間還原、音訊回接、輸出探測與幀數／時長驗證。
- M2：區間共用色盤、色塊輪廓、孔洞、獨立暗線遮罩、純向量 SVG、版本化 baseline 專案、精確重複影格資產重用。
- M2：SVG 透過 resvg 真正重建後製作 MP4，並量測重建誤差，沒有直接把色盤量化點陣圖當成向量渲染結果。
- M2：本地 HTML 檢視器、逐幀、SVG 縮放、原始／重建抽樣對照圖。
- M4 前置：自行實作 orientation 精確回退及簡單多邊形 ear clipping，尚未接入色塊管線。

## 實驗素材

`Isekai.mkv`：91.010 秒、1920×1080、H.264、24000/1001 FPS、時間基準 1/1000、FLAC 48 kHz 日語雙聲道。原檔 181,096,011 bytes，未修改。

全片縮圖：`output/isekai-analysis/contact-sheet.jpg`。選取兩段具有不同內容的片段，不代表完整測試矩陣：

| 片段 | 選取理由 | 實際來源起點 | 輸出時長 | 幀數 | 不重複 SVG 資產 |
| --- | --- | --- | --- | --- | --- |
| 1–6 秒 | 角色近景、鏡頭拉遠、細線與複雜景物 | 1.001 秒 | 4.999 秒 | 120 | 120 |
| 20–25 秒 | 天空漸層、切鏡、機械結構與深色區域 | 20.020 秒 | 4.980 秒 | 120 | 106 |

共同設定：640×360、32 色、epsilon 0.65 像素、暗線閾值 65、隨機種子 7。

```powershell
.venv/Scripts/python -m vecanima vectorize Isekai.mkv --start 1 --duration 5 --width 640 --colors 32 --out output/isekai-character-v2
.venv/Scripts/python -m vecanima vectorize Isekai.mkv --start 20 --duration 5 --width 640 --colors 32 --out output/isekai-baseline-v2
```

以上實驗目錄已存在；重跑請改輸出名稱。`v2` 是像素格邊界修正後版本。較早 `output/isekai-baseline-20s` 保留供比較，不是目前建議檢視版本。

## 本輪量測

| 片段 | RGB 通道平均絕對誤差（0–255） | 逐幀 PSNR 平均 | 不重複 SVG bytes | 總處理時間 |
| --- | --- | --- | --- | --- |
| 1–6 秒 | 4.8569 | 30.4604 dB | 70,268,615 | 48.32 秒 |
| 20–25 秒 | 4.4708 | 32.4714 dB | 51,053,239 | 39.66 秒 |

時間為本機執行結果，包含抽幀、色盤、向量化、SVG 渲染與編碼，不能推論其他硬體速度。數值比較對象是工作解析度的 FFmpeg 原始影格，而不是原始 1080p。PSNR 為逐幀 dB 算術平均。SVG bytes 不包含幾何 JSON、PNG、MP4，因此不是專案總大小。

舊輪廓基準在 20–25 秒平均 MAE 5.8954 / PSNR 28.7046 dB；改成像素格邊界後 MAE 4.4708 / PSNR 32.4714 dB，但 SVG 由 38,647,537 增加至 51,053,239 bytes。品質提升有資料量成本，仍需要正式曲線與網格模型。

逐幀資料見各實驗 `report.json`。`near_static_delta_mae` 只是未做運動補償的近靜態像素診斷，不能當成正式動畫閃爍指標。尚未測得跨幀穩定改善，因為該模組尚未實作。

## 驗證結果

`python -m pytest -q`：**19 passed**。

- 幾何：凸／凹多邊形、順逆時針、共線中間點、相鄰重複點、狹長形狀、面積守恆與三角形方向。
- 非法幾何：自交、退化、非相鄰重複點、非有限座標，明確拒絕。
- 近共線 orientation 對照精確有理數。
- SVG：孔洞、單像素線、全黑圖、對角接觸、巢狀孔洞、角落像素與重現性。
- 媒體：VFR PTS、非零來源起點、非幀邊界 seek、有音訊／無音訊、輸出幀數、持續時間與音訊起訖。
- 不覆寫非空輸出。

實際 Isekai 兩段皆驗證預覽幀數等於 120，影片時長與時間軸相符；音訊可用。已目視檢查四組抽樣對照。瀏覽器檢查 SVG 成功載入、下一幀顯示 2/120（0.042s）、縮放控制可調整。

`scripts/setup.ps1` 已以鎖定依賴重跑，環境診斷通過。

## 已知限制與下一個開發批次

1. 暗線遮罩會將大片暗色填色誤認為筆觸，且只用單一代表色；下一批實作局部剖面、中心線與左右線寬。
2. 色盤量化造成漸層色階，複雜背景細節流失；需要共享邊界、CDT、網格顏色與細化。
3. 多邊形輪廓放大後可見階梯／尖角；需要 Bézier 擬合與誤差控制。
4. 只有完全重複畫面共用資產；沒有切鏡分段、運動補償、物件追蹤或時間濾波。共用區間色盤不是跨幀追蹤。
5. SVG 與幾何 JSON 目前很大，沒有完成壓縮或差量編碼。
6. 暫未支援原生網格 WebGL 播放、HDR tone mapping、多音軌選擇與字幕燒錄。
7. ear clipping 目前只支援簡單無孔多邊形，不是 CDT；完整受約束 Delaunay 必須繼續自行實作，沒有 CGAL。
8. 尚未完成長片串流、斷點續跑或跨階段快取；輸出失敗會標記 incomplete 並保留診斷。

Rust 移植完成基準後，依計畫進行 M3 正式筆觸模型，並建立完整合成資料集；接著 M4 三角化／漸層，再做 M5 時間穩定。

## 新模型進度

本文件前段為歷史基準，最新 M3–M7 實作及檢查狀態見 `docs/IMPLEMENTATION.md` 的 2026-10-08 本輪進度。已實作筆觸剖面、自行 CDT、原生網格漸層、筆觸時間追蹤、WebGL 播放、通用輸入及可校驗快取／續跑；歷史「尚未實作」敘述不代表目前程式。尚未完成全項最終驗收，不宣稱論文完整重現。

## 最新通用品質整合

上段為歷史基準；正式 Rust 已有區域與筆觸時間關聯、自行精確 CDT、共享平滑界面、可變寬變色筆觸、線性光網格、WebGL 播放及可校驗快取／續跑。本輪另完成強弱證據連通、乾淨 donor 調和恢復及連續 SVG 漸層；43 Rust／28 Python／嚴格 Clippy 通過。三個 640×360 細節場景完成獨立稽核，全時長 2180 幀驗收中。最新狀態與完整度量統一見 ACCEPTANCE.md，計畫 GENERAL-QUALITY-PLAN.md，錯誤 ERRORS.md。
