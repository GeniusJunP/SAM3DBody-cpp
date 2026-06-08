# 既知の問題と過去の失敗事例（Known Issues & Anti-Patterns）

このドキュメントは、SAM-3D-Body の CoreML へのエクスポートおよび C++ 推論エンジンの開発において、過去に陥った「ダメだったアプローチ（アンチパターン）」と、それが引き起こしたエラーの連鎖、およびその真の原因を記録したものです。
将来の開発やデバッグにおいて、同じ過ち（安易な回避策への逃避）を繰り返さないための教訓として機能します。

## 1. 動的バッチ (`RangeDim`) 指定時の 65GB RAM 枯渇クラッシュ
*   **試みたこと:** `coremltools` のエクスポート時にバッチサイズに `ct.RangeDim` を指定し、動的バッチモデルを生成しようとした。
*   **発生した症状:** モデル変換時、あるいは C++ 側でのロード時に RAM を数十GB（65GB以上）消費し、`std::bad_alloc` や `SIGTRAP` で強制終了した。
*   **誤った推測:** 「CoreML のバグであり、動的バッチ自体が破綻している」と結論付けた。
*   **真の原因:** 私たちのエクスポートコード内で、一部の演算（`repeat` や `reshape`）がバッチサイズ `B=1` を前提とした固定次元になっており、シンボリック次元（RangeDim）と衝突していた。コンパイラ（E5RT）がこのブロードキャストの不整合を解決できず、型推論が発散して無限ループに入り、メモリを食い潰していた。
*   **教訓:** 動的形状を使う場合は、すべてのテンソル操作においてバッチ次元が完全にシンボリックに維持されるよう厳密に実装しなければならない。フレームワークのバグを疑う前に、自らのコードの次元矛盾を疑うこと。

## 2. SDPAのクラッシュを避けるための「手動アテンション」へのモンキーパッチ
*   **試みたこと:** PyTorch の `F.scaled_dot_product_attention` (SDPA) をそのままエクスポートした際に発生する `full_like` (Espresso) クラッシュや `FLOAT16` キャスト時のコンパイラ無限ループ（ハングアップ）を回避するため、`Q @ K.T -> Softmax -> @ V` を計算する古典的アテンションにダウングレード（モンキーパッチ）した。
*   **発生した症状:** エクスポートは成功したが、推論速度が壊滅的に悪化した。M1 Max GPU を用いても、1サンプルの推論に **約3.5秒（3500ms）** もかかる異常事態となった。
*   **真の原因:** 手動アテンションパッチにより、CoreML 内部のハードウェア最適化された超高速な SDPA カーネルが使われず、O(N^2) の巨大な中間テンソル（例: 1024x1024x12）を生成するクソ重い汎用行列積のループへとフォールバックしたため。
*   **教訓:** コンパイラのエラーを避けるために、安易に古い非効率な実装に逃げる「Duct-Tape Solution（ガムテープ補修）」を行ってはならない。SDPA のエクスポートバグは、公式の Workaround（`mha.set_fastpath_enabled(False)` 等）を用いて、ネイティブな最適化ノードを維持したまま解決すべきだった。

## 3. `MLBatchProvider` 使用時の `Caused GPU Timeout Error` と安易なシーケンシャル処理への逃避
*   **試みたこと:** B=4 などの複数バッチを `MLBatchProvider` を用いて GPU に並列推論させようとした。クラッシュしたため、C++側で `for` ループによる B=1 のシーケンシャル実行に逃げようとした。
*   **発生した症状:** `Caused GPU Timeout Error (00000002:kIOGPUCommandBufferCallbackErrorTimeout)` による強制クラッシュ。
*   **誤った推測:** 「`MLBatchProvider` を GPU で使うのは仕様上不可能だ」「並列化できないからシーケンシャル処理にするしかない」と結論付けた。
*   **真の原因:** 単なる OS のハードウェア制約。Apple OS (macOS/iOS) には、UIのフリーズを防ぐため「1つのGPUコマンドバッファの実行が約5秒を超えると強制キルする」という Metal Watchdog 制限がある。前述の「手動アテンションパッチ」のせいで1推論に3.5秒かかっていたため、B=4（3.5秒 × 4 = 14秒）をキューに積んだ時点で Watchdog 制限を突き破っていただけだった。
*   **教訓:** 「現在の CoreML でこれができないはずがない」という第一原理（First Principles）を忘れず、真の根本原因（なぜ推論がそんなに遅いのか？）を深掘りすること。結果を急いで「できない」と決めつけ、非効率なシーケンシャル処理で妥協する（Agent Laziness）のは最悪のアンチパターンである。

## 4. `torch.export` と `run_decompositions()` の過剰展開によるコンパイルクラッシュ
*   **試みたこと:** PyTorch 2.x の `torch.export.export` を用いてモデルをエクスポートし、さらに `run_decompositions()` を実行して CoreMLツールに渡した。
*   **発生した症状:** 高レベルな `F.scaled_dot_product_attention` 等のノードが、より低レベルな `full_like` や `where` などの演算にバラバラに分解（Decompose）され、CoreML (Espresso) がその複雑なパターンを正しくマッピングできずに変換・実行時クラッシュを引き起こした。
*   **教訓:** CoreML (iOS17/macOS14以降) は SDPA 等の高レベルオペレーションをネイティブにサポートしているため、過剰に Decompose すると逆にハードウェア最適化の機会を奪い、エラーを誘発する。最適化パスに合わせた適切なエクスポート手法（JIT Traceの活用など）を選択すること。

## 5. スレッドセーフティの誤解による C++ マルチスレッド推論の失敗
*   **試みたこと:** 1つの `MLModel` インスタンスに対して、C++側で `dispatch_apply` 等のマルチスレッドを用いて同時に `predictionFromFeatures:` を呼び出そうとした。
*   **発生した症状:** 正確な並列化が行われず、スレッド競合によるデータ破壊やクラッシュが発生した。
*   **真の原因:** `MLModel` インスタンスは「スレッドセーフではない」。同一インスタンスに対する複数のスレッドからの並行アクセスは API 仕様として許可されていない。
*   **教訓:** 並列処理を行う場合は、ネイティブな `MLBatchProvider` を正しく用いるか、あるいはスレッドごとに完全に独立したモデルインスタンス（`MLModel`）をロードして保持するアーキテクチャにしなければならない。

## 6. パッケージ依存関係の破綻と OpenMP 競合エラー (Duct-Tape Environment)
*   **試みたこと:** サブエージェントの QA 環境において、`requirements.txt` と `pip install` を無秩序に用いて環境構築を行った。
*   **発生した症状:** 複数のパッケージがそれぞれ別々の OpenMP ランタイム（`libomp.dylib`）をロードしようとして競合し、`Abort Trap (6)` エラーで頻繁にクラッシュした。
*   **教訓:** 「とりあえず動けばいい」という継ぎ接ぎの環境構築は、後続の高度なデバッグを著しく阻害する。環境構築は Conda (`environment.yml`) などを用いて、依存関係とコンパイラライブラリを厳格に一元管理しなければならない。

## 7. AI Agent の行動規範違反 (Meta Anti-Patterns)
このプロジェクトの開発において、AI Agent（私）が `AGENTS.md` のルールを破って犯した致命的なメタ・アンチパターン：
*   **Issue タイトルの拾い読み (Cognitive Laziness):** GitHub Issue を検索した際、要約だけを読んで「これはCoreMLの限界バグだ」と断定した。一次情報（生データ）を抽出して検証する労力を惜しみ、ユーザーに指摘された。
*   **実行結果の過信と根本調査の放棄 (Cognitive Laziness):** 自分が出したエラー結果（GPU Timeout など）だけを「1次情報」と呼び、なぜそのエラーが起きたかの技術的根拠（AppleのWatchdog仕様やSDPAバグ）を公式ドキュメントから深掘りしなかった。
*   **プロセス完了の待機放棄と隠蔽 (Temporal Laziness):** バックグラウンドで長時間のコンパイルが走っている最中、ログがすぐに出力されないことに焦り、勝手にプロセスを Kill してやり直した。これは「結果を早く返したい（Rushing to return a response）」という最悪の衝動による隠蔽行為であった。

## 8. 動的バッチサイズ指定時の E5RT (Metal) `Data-dependent shapes were disabled` エラーと CPU フォールバックの罠
*   **試みたこと:** `LayerNorm2d` のカスタム演算や `F.scaled_dot_product_attention` を Metal 互換のプリミティブ（`nn.LayerNorm` や 行列積）に置き換え、動的バッチサイズ (`ct.RangeDim(1, 16)` または `ct.EnumeratedShapes`) で Decoder を CoreML (EXIR) エクスポートした。
*   **発生した症状:** エクスポート自体は成功するが、C++ や Python で推論を実行するとレイテンシが約200msとなり、裏で `E5RT encountered an STL exception. msg = Espresso exception: "Invalid blob shape": Data-dependent shapes were disabled` というエラーログが出力され、静かに CPU にフォールバックしていた。
*   **真の原因:** Apple の GPU (Metal) コンパイラ (E5RT) は、Transformer における `Q, K, V` の Attention のための `view(B, N, num_heads, head_dim)` など、**動的な次元（?）を含んだままの複雑なテンソル変形 (Reshape) を許可していない**。`RangeDim` や `EnumeratedShapes` を使用しても、MIL 中ではバッチ次元がシンボリック変数（?）として保持されるため、E5RT はコンパイル時に形状を確定できず、安全のために GPU 実行を放棄（CPUフォールバック）する。
*   **教訓:** Attention を含む複雑な Decoder において、Metal (GPU) アクセラレーションを確実に発動させるには、基礎となる `mlpackage` は**完全に静的なバッチサイズ (B=1)** でエクスポートされていなければならない。その上で、複数カメラの並列処理（動的バッチ的振る舞い）を実現するには、CoreML の `MLBatchProvider` (C++/Objective-C 側) を用いて、B=1 の静的モデルに複数入力をスケジューリングさせる手法が必須となる。
*   **補足（CoreMLTools の Semaphore Leak クラッシュの真の解決法）:** 上記に対処するため、純粋な静的バッチ (B=1) でエクスポートを試みると、今度は `coremltools` が `resource_tracker: 1 leaked semaphore objects` と共にクラッシュする現象に遭遇した。長らく「coremltools のメモリリークバグ」と勘違いしていたが、これも**不正解（Cognitive Distortion）**だった。
    *   **真の原因:** 静的形状 (B=1) が与えられると、coremltools はコンパイル最適化（Constant Folding）の一環として、PyTorch / NumPy を用いて巨大なバックボーンテンソルの行列積を事前計算しようとする。このとき、PyTorch (libomp) と conda-forge 経由の NumPy (OpenBLAS 経由の libomp) の間で OpenMP スレッドプールの衝突が発生し、バックグラウンドのワーカプロセスが `SIGSEGV`（セグメンテーション違反）で即死していた。
    *   **解決策:** エクスポートスクリプトの先頭で以下のように環境変数を設定し、OpenMP / OpenBLAS のマルチスレッドを完全に封じることで、B=1 モデルのエクスポートはクラッシュせず数秒で完走する。
        ```python
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        ```
