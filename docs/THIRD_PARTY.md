# 第三方來源與演算法紀錄

## 執行環境

實際版本以 `uv.lock` 與 `scripts/ffmpeg-lock.json` 為準；依賴原始授權隨安裝套件保留。

| 元件 | 本輪版本 | 來源與用途 | 授權紀錄 |
| --- | --- | --- | --- |
| OpenCV 原生 SDK | 4.14.0 | 使用者提供 `third_party/opencv-4.14.0.zip`，本地編譯供 Rust 使用；SHA256 見 `scripts/opencv-lock.json` | 原始碼 LICENSE、COPYRIGHT 與內附第三方授權保留於 `.tools/opencv-src/opencv-4.14.0/` |
| opencv Rust 綁定 | 0.101.0 | https://github.com/twistedfall/opencv-rust | MIT；crate 原始碼與授權保留於 `third_party/rust/` |
| resvg Rust 渲染器 | 0.48.1 | https://github.com/linebender/resvg ，原生 SVG 重建與量測；關閉預設文字／字型／點陣圖片功能 | 依發佈 crate 中 LICENSE 保留於 `third_party/rust/`；遞移依賴亦保留各自授權 |
| FFmpeg / ffprobe | 9.0.2 Gyan essentials | https://ffmpeg.org/download.html 列出的 Windows 建置提供者：https://www.gyan.dev/ffmpeg/builds/ | 此建置為 GPLv3；原授權與 README 保留於 `.tools/` |
| NumPy | 2.5.3 | https://numpy.org/ ，數值與資料處理 | wheel 中 BSD-3-Clause 與附帶元件授權，見安裝包 LICENSE |
| OpenCV contrib headless | 4.14.0.94（cv2 4.14.0） | https://github.com/opencv/opencv-python ，影像處理 | Apache 2.0 與附帶元件授權，見 wheel 的 LICENSE / LICENSE-3RD-PARTY |
| resvg-py | 0.5.0 | https://github.com/baseplate-admin/resvg-py ，獨立 SVG 參考渲染 | Python 綁定 MIT；底層 resvg 與依賴授權依各自發佈檔保留 |
| pytest | 9.1.1 | https://github.com/pytest-dev/pytest ，測試 | MIT；僅開發依賴 |
| SciPy | 1.18.1 | https://scipy.org/ ，數值最佳化與空間查詢 | BSD-3-Clause 與 wheel 附帶元件授權 |
| scikit-image | 0.26.0 | https://scikit-image.org/ ，骨架與品質量測 | BSD-3-Clause 與 wheel 附帶元件授權 |
| hatchling | 1.32.4 | https://github.com/pypa/hatch ，本專案建置及離線 editable 安裝 | MIT；開發／建置依賴 |

本輪另下載上述元件的完整遞移依賴（包含 Pillow、imageio、NetworkX 等）。精確下載 URL、版本、SHA256 與各 wheel 授權位置見 `third_party/manifest.json`；本地安裝說明見 `third_party/README.md`。本地 uv 執行檔來源為此機既有 uv 安裝，複製後記錄版本與 SHA256。

FFmpeg 使用獨立子程序呼叫。尚未發佈安裝包；後續打包時需按實際隨包元件附授權、來源與對應義務。本清單不取代各原始授權。

## 自行實作幾何

未安裝或連結 CGAL，也未複製 CGAL 程式碼。

1. David Eberly, *Triangulation by Ear Clipping*, Geometric Tools：
   https://www.geometrictools.com/Documentation/TriangulationByEarClipping.pdf
   - `geometry.triangulate_simple` 採 ear clipping 概念獨立撰寫。
   - 目前限單一簡單多邊形，不支援孔洞或 CDT；非法輸入明確拒絕。
2. Jonathan Richard Shewchuk, *Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates*, 1996 technical report / 1997 publication：
   https://www.cs.cmu.edu/~quake/robust.html
   - 借鑑近退化幾何需要穩健符號判斷的設計原則。
   - 本版 `orient` 使用浮點快速路徑與 Python Fraction 精確有理數回退，不是原論文的 expansion arithmetic 實作，也未複製其原始碼。
   - incircle、CDT 約束恢復、孔洞與網格細化尚待實作及驗證。

Python `vectorize.mask_rings` 與 Rust `src/vectorize.rs::mask_rings` 的像素格邊界追蹤由本專案撰寫，沿前景／背景交界的格邊走訪，對角接觸優先右轉，保留四連通元件；SVG 使用 evenodd 填色表達孔洞。Rust 使用方向位元與排序起點，沒有呼叫第三方三角化；輪廓簡化使用 OpenCV approxPolyDP。

## 本地 AI 邊界模型

新增 AniLines basic ONNX 的 Rust/OpenCV DNN 原生推論。官方程式與 basic／detail 權重、MIT LICENSE、來源及 SHA256 保存在 third_party/anilines；detail 僅比較。PyTorch／ONNX 僅作轉換工具，wheel 及授權本地封存。詳見 third_party/anilines/LOCAL-SETUP.md 和 manifest.json、tools-manifest.json。

原作者：https://github.com/zhenglinpan/AniLines-Anime-Lineart-Extractor 。權重來源為作者提供的 https://huggingface.co/spaces/aidenpan/AniLines-Anime-Lineart-Extractor/tree/main/weights 。另查閱 Anime2Sketch 及 LineDistiller，沒有整合；後者官方模型標示 CC BY-NC 4.0，沒有假設其授權為 MIT。

## 正式 Rust 幾何與恢復（覆蓋上段 Python 基準限制）

`src/geometry.rs` 已自行實作浮點快速判斷與 exact dyadic bigint orientation／incircle、增量 Delaunay、約束 cavity 恢復、孔洞及退化輸入驗證；`shared.rs` 共用平滑界面、`mesh_fill.rs` 線性光網格與誤差細化。沒有安裝、連結或複製 CGAL／Triangle 三角化程式碼，bigint 只作精確數值表示。

`fill_recovery.rs` 的 screened harmonic 是自行建立帶正 screen 的離散 SPD 線性系統，乾淨 donor 為 Dirichlet 邊界，真實色跳變使用無通量界面；Jacobi 預條件共軛梯度在 Rust 實作。不是 CGAL 或第三方 inpainting 呼叫，收斂與實測契約見 ACCEPTANCE.md。
