# Rust CLI 開發與依賴

目標程式為原生 Rust CLI。Python 只保留作為演算法參考、測試素材生成、依賴下載與品質比對工具；正式 CLI 不以 Python 子程序代跑核心處理。

## 依賴

- Rust stable x86_64-pc-windows-msvc（此機既有 rustc 1.97.1 / cargo 1.97.1）。
- 此機既有 Visual Studio MSVC 及 LLVM / libclang。
- `opencv` crate 0.101.0，來源：https://github.com/twistedfall/opencv-rust 。
- OpenCV 4.14.0：使用者提供的 `third_party/opencv-4.14.0.zip` 原始碼，固定 SHA256 見 `scripts/opencv-lock.json`；由此機 MSVC 編譯本地 SDK。
- FFmpeg / ffprobe，版本鎖與本地 ZIP 見 `scripts/ffmpeg-lock.json`。
- clap、anyhow、serde、serde_json；全部遞移 Rust 依賴依 Cargo.lock 鎖定，原始碼 vendor 在 `third_party/rust/`。
- resvg 0.48.1（https://docs.rs/resvg/0.48.1/resvg/）：Rust 內獨立渲染 SVG；關閉文字、系統字型與點陣圖片等預設功能。

只啟用 OpenCV imgcodecs、imgproc、dnn 與 clang-runtime；core 自動包含。需要其他模組時再明確開啟，不編譯所有預設模組。官方 Windows SDK 與 Python OpenCV contrib wheel 是不同產品；不能將 cv2 wheel 當成 Rust 的原生 SDK。

## 安裝與建置

此工作區的 Rust crate 已下載；以下指令校驗本地 ZIP、解壓並編譯原生 SDK 至 `.tools/opencv-sdk/`，不下載另一版 OpenCV：

```powershell
.venv/Scripts/python scripts/install_opencv.py --jobs 8
./scripts/cargo.ps1 build --locked --offline
./scripts/cargo.ps1 test --locked --offline
./scripts/cargo.ps1 -CargoArguments @('clippy', '--locked', '--offline', '--all-targets', '--', '-D', 'warnings')
./scripts/cargo.ps1 run --locked --offline -- doctor
```

原始碼在 `.tools/opencv-src/opencv-4.14.0/`，CMake 建置在 `.tools/opencv-build/`。使用既有 CMake、Visual Studio 18 2026、x64 Release；分別編譯 core、imgproc、imgcodecs、dnn 共享 DLL。PNG、JPEG、zlib 使用 ZIP 內附原始碼，關閉 IPP、ITT、OpenCL 與內建影片後端；影片解碼由獨立 FFmpeg 負責。未啟用 contrib 模組，後續按演算法需求增添。重跑會重新校驗 ZIP 並增量建置。

`scripts/cargo.ps1` 使用明確的 OpenCV include / lib / DLL 路徑，並從既有 clang 安裝尋找 libclang.dll。環境變數只影響呼叫它的 PowerShell 程序，不修改登錄或系統設定。若 libclang 不在 clang.exe 同目錄，先設定 `$env:LIBCLANG_PATH`。

Rust 原始依賴存檔可重新建立：

```powershell
cargo fetch --locked
cargo vendor --locked --offline --versioned-dirs third_party/rust > .cargo/config.toml
```

vendor 內保留原 crate 原始碼、授權與 `.cargo-checksum.json`。`.cargo/config.toml` 將 crates.io 解析替換為本地來源，`--offline --locked` 防止建置時網路下載或自動更動版本。Rust / MSVC / libclang 工具鏈使用此機既有安裝，未打包成可攜編譯器。

## 起始指令

```powershell
./scripts/cargo.ps1 run --locked --offline -- probe Isekai.mkv
./scripts/cargo.ps1 run --locked --offline -- extract Isekai.mkv --start 20 --duration 5 --width 640 --out output/rust-extract
./scripts/cargo.ps1 run --locked --offline -- edges output/isekai-character-v2/original/000001.png --out output/rust-edges.png
./scripts/cargo.ps1 run --release --locked --offline -- vectorize Isekai.mkv --start 1 --duration 5 --width 640 --colors 32 --out output/new-vectorization
```

`doctor` 真正執行 OpenCV RGB 至灰階轉換；`edges` 真正執行 OpenCV Canny；`probe` / `extract` 使用 Rust `std::process::Command` 直接呼叫 FFmpeg，保留解碼實際時間戳並拒絕覆寫非空輸出。

`vectorize` 在 Rust 內完成共用 Lab 色盤、形態學 black-hat 細線殘差、連通骨架、多尺度 Hessian 中心修正、次像素法線剖面、骨架交界連續配對、平滑 Bézier、節點雙側線寬與線性 RGB 保形三次內插。正式 ribbon 使用每個節點的顏色與線寬，平均色只留作摘要。填色以筆觸覆蓋及鄰近的來源弱殘差證據移除線條與抗鋸齒殘邊。

填色 `--fill-model flat` 使用共用平滑閉合輪廓；`--fill-model mesh` 使用自行實作 exact dyadic predicates、Delaunay／約束 cavity／孔洞及線性光三角網格，沒有 CGAL。num-bigint 只用於精確算術。SVG 以漸層／細分三角形純向量近似，沒有嵌入圖片；原生 CPU 四子像素積分、WebGL MSAA，以及 SVG 扇區統一裁切處理邊緣。

```powershell
./scripts/cargo.ps1 run --release --locked --offline -- image input.png --fill-model mesh --out output/image
./scripts/cargo.ps1 run --release --locked --offline -- sequence frames.json --temporal --out output/sequence
./scripts/cargo.ps1 run --release --locked --offline -- vectorize Isekai.mkv --start 1 --duration 1 --width 640 --fill-model mesh --mesh-vertices 48000 --temporal --out output/smooth
./scripts/cargo.ps1 run --release --locked --offline -- serve output/smooth --port 51284
```

序列 manifest 為 `[ {"path":"a.png","duration":0.04}, {"path":"b.png","duration":0.12} ]`；影像路徑相對於 manifest，所有影格工作尺寸需一致。`--no-preview` 略過 MP4，仍輸出播放器及品質報告。

主要參數可透過各子命令 `--help` 查看：`--stroke-contrast`、`--stroke-max-width`、`--stroke-min-aspect`（〔中心線長 + 線寬〕／線寬，以 90 百分位總線寬估計並納入端帽，預設 2）、`--curve-error`、`--stroke-color-error`、`--stroke-gap-max`；`--mesh-error`、`--mesh-vertices`、`--mesh-spacing`、`--mesh-boundary-contrast`、`--svg-color-error`；以及 `--temporal`。設定保存於輸出，不依 Isekai 的角色、座標或顏色分支。

`--temporal` 使用自行金字塔 Lucas–Kanade／前後一致性及 robust affine 相機估計，記錄筆觸／區域對應與假設事件，按運動殘差濾波線寬。沒有把幾何事件當成語意辨識或宣稱時間穩定已完整驗收。

非空輸出預設拒絕覆寫。同來源／設定／實作可加 `--resume`；SHA256 檢查來源及階段快取，損壞重新計算，原子狀態保留失敗原因與輸出內 `ERRORS.md`。時間濾波從未濾波快取重播。實作變更會拒絕混用舊輸出，請使用新目錄。

瀏覽器播放器在 `http://127.0.0.1:51284/`，提供筆觸／色塊 checkbox、原生／SVG、原圖對照、逐幀／縮放、節點線寬顏色檢視。`serve` 限本機只讀，支援影音範圍請求。大網格使用二進位 f32 GPU 緩衝；JSON 保留完整可編輯模型。

## 驗證

```powershell
./scripts/cargo.ps1 test --locked --offline
./scripts/cargo.ps1 -CargoArguments @('clippy','--locked','--offline','--all-targets','--','-D','warnings')
./scripts/cargo.ps1 build --release --locked --offline
.venv/Scripts/python -m pytest -q
```

本次 43 個 Rust 單元測試、28 個 Python 驗證測試（含 9 個原生 CLI 端到端）與嚴格 Clippy 通過。涵蓋精確謂詞、任意方向約束、孔洞、斜線／變色／變線寬、密集交界、曲線回折、SVG 裂縫與色差、跨不同底色的筆觸分離、淡邊與真實陰影、AI 弱線連通及幻線拒絕、相鄰陰影線寬、VFR／音訊／路徑／奇數尺寸、模型內容雜湊、快取損壞及續跑逐位元一致。Python 只作素材生成與驗證。

完整全片品質、資料量、時間穩定與效能驗收仍待完成，見 IMPLEMENTATION.md。SVG 資料可能大於原片，輸出明確報告網格預算與未滿足色差，不能僅以 PSNR 當作外觀驗收。

新增 crate 時暫時停用本地來源替換，fetch／vendor 後還原；授權與 checksum 保留於 third_party/rust。依賴包含 sha2 及 num-bigint，全部版本依 Cargo.lock 鎖定。

## 新增本地模型與乾淨填色

SDK 建置列表現在包含 dnn，啟用來源 ZIP 內的 protobuf；scripts/install_opencv.py 與 scripts/cargo.ps1 已同步。新增 `--line-model`（可選 AniLines basic RGB ONNX）及 `--fill-edge-padding`（預設 2 工作像素）。模型以內容 SHA256 納入 run／stage 身分，變更權重不能續用舊向量快取。

`lineart input.png --model third_party/anilines/basic.onnx --out output/lineart.png` 輸出原生模型證據；`render-svg output/project/frames/000001.fills.svg --out output/fills.png` 檢查純填色 SVG。兩者皆拒絕覆寫已有輸出。正式程式不需要 PyTorch；轉換及離線模型工具見 ../third_party/anilines/LOCAL-SETUP.md。

最新 43 Rust／28 Python（含 9 原生 CLI）／嚴格 Clippy 通過。最新播放器 `output/general-quality-svg`；舊輸出屬歷史實驗。來源線寬受 AI 支撐約束或推估時標記 inferred／低信心；色彩仍取來源，填色未知區必須有來源暗谷支撐。見該輸出的 FEATURE-CHECK.md。

## 通用恢復與獨立輸出稽核

筆觸採強弱模型支撐連通，但必須通過來源暗谷觀測；顏色從來源墨谷獨立取樣。只有雙側有效錨點、沿模型支撐且有來源殘差的缺失剖面才內插，推估標記保留。填色採乾淨核心 donor 與 screened harmonic 線性光恢復；真實色跳變施加無通量障壁，避免最近 donor 分區生成暗斑或漸層斷帶。

網格硬界面要求來源局部跳變，不能單靠色盤分類。SVG 筆觸及色塊共用連續漸層，`--svg-color-error` 預設 2；輸出資料量因此增加。幾何 JSON 保存 `harmonic_iterations`、`harmonic_residual`、`harmonic_converged` 及 `suppressed_palette_boundaries`，不足預算不隱藏。

```powershell
.venv/Scripts/python.exe scripts/verify_project.py output/general-quality-svg --model third_party/anilines/basic.onnx
.venv/Scripts/python.exe scripts/verify_project.py output/acceptance-general-320 --minimum-duration 90 --model third_party/anilines/basic.onnx
```

稽核僅接受 status 為 complete 的輸出，逐檔 SHA256 與逐幀幾何／時間／影音檢查寫入 acceptance-audit.json；不執行向量化。已知相機平移合成驗證提供外部真值，實片匹配殘差只作描述。結果、體積與能力限制見 ACCEPTANCE.md。
