# light-torchSISSO

[![PyPI version](https://badge.fury.io/py/light-torchsisso.svg)](https://badge.fury.io/py/light-torchsisso)


**`light-torchSISSO` は、シンボリック回帰アルゴリズム SISSO (Sure Independence Screening and Sparsifying Operator) を、メモリ効率と計算効率を重視してPythonで再実装した軽量なライブラリです。**

オリジナルのSISSOが持つ、データから解釈可能な数式モデルを発見する能力を維持しつつ、以下の特徴を持っています。

-   **🧠 メモリ効率の良いアーキテクチャ**: 「レシピ化」アーキテクチャにより、特徴拡張（Feature Expansion）時のメモリ消費を劇的に削減。大規模な特徴空間の探索を可能にします。
-   **⚡ 高速な探索**:
    -   **レベルワイズSIS**: 特徴生成とスクリーニングを段階的に行うことで、無駄な計算を削減し、探索を高速化します。
    -   **2つの探索戦略**: 最適解を保証する総当たり探索 (`exhaustive`) と、高速なヒューリスティック探索 (`lasso`) を選択可能です。
-   **🧩 scikit-learnライクなインターフェース**: `fit()` / `predict()` という直感的なAPIを提供し、既存の機械学習ワークフローに簡単に組み込めます。
-   **🔩 安定した計算**: 内部の線形回帰計算に実績のある`NumPy`を使用し、数値的な安定性を確保しています。

このライブラリは、物理法則の発見、材料科学における物性予測、金融モデリングなど、データ背後に潜むメカニズムを解釈可能な数式の形で明らかにしたいあらゆる分野で活用できます。

## 📥 インストール

`pip`を使ってGitHubリポジトリから直接インストールできます。

```bash
pip install git+https://github.com/shinkawa-shuan/light-torchsisso.git
```

もしパッケージをアップデートしたい場合は、`--upgrade`オプションを追加してください。

```bash
pip install --upgrade git+https://github.com/shinkawa-shuan/light-torchsisso.git
```

## 🚀 クイックスタート

わずか数行のコードで、データから数式モデルを学習し、予測を行うことができます。

```python
import pandas as pd
import numpy as np
from light_torchsisso.model import SissoModel

# 1. ベンチマークデータの準備
# 真の式: y = 2 * sin(f0) + f1^2 + ノイズ
np.random.seed(42)
X1 = np.random.rand(100) * 2
X2 = np.random.rand(100) * 3
y_train = 2 * np.sin(X1) + X2**2 + np.random.randn(100) * 0.1
df_train = pd.DataFrame({"target": y_train, "f0": X1, "f1": X2})

# 2. SissoModelのインスタンスを作成
model = SissoModel(
    n_expansion=2,          # 特徴拡張のレベル
    n_term=2,               # 探すモデルの最大項数
    operators=["+", "sin", "pow2"], # 使用する演算子
)

# 3. モデルの学習
model.fit(df_train)

# 4. 学習結果の確認
print(f"発見された数式: {model.equation_}")
print(f"RMSE: {model.rmse_:.4f}")
print(f"R2スコア: {model.r2_:.4f}")

# 5. 新しいデータで予測
df_test = pd.DataFrame({"f0": np.array([0.5, 1.0]), "f1": np.array([1.0, 2.0])})
predictions = model.predict(df_test)
print(f"\n予測結果: {predictions}")
```

**出力例**:
```
... (学習ログ) ...
Best Model Found (2 terms):
  RMSE: 0.096579
  R2:   0.998821
  Equation: +1.008824 * ^2(f1) +1.971188 * sin(f0) +0.003702

発見された数式: +1.008824 * ^2(f1) +1.971188 * sin(f0) +0.003702
RMSE: 0.0966
R2スコア: 0.9988

予測結果: [3.0481014 5.6796246]
```

## 🛠️ 使い方ガイド

### `so_method`: 探索戦略の選択

`SissoModel`は2つのモデル探索戦略（Sparsifying Operator）を提供します。

#### 1. `exhaustive` (デフォルト)
候補となる特徴のすべての組み合わせをテストする総当たり探索です。
-   **長所**: 最適な組み合わせを見つけられる可能性が最も高い。シンプルで解釈しやすいモデルが見つかりやすい。
-   **短所**: 候補特徴や項数が増えると、計算時間が爆発的に増加する。

```python
model = SissoModel(
    n_term=3,
    so_method="exhaustive", # デフォルトなので省略可
    operators=["+", "-", "*", "sqrt"]
)
```

#### 2. `lasso`
Lasso回帰を用いて、多数の候補から重要な特徴を高速に選択します。
-   **長所**: 非常に高速。`exhaustive`では現実的でない大規模な探索空間でも有効なモデルを見つけられる可能性がある。
-   **短所**: `alpha`パラメータの調整が必要。見つかるモデルが複雑になりがちで、必ずしも最適解とは限らない。

```python
model = SissoModel(
    so_method="lasso",
    alpha=0.01, # Lassoの正則化パラメータ。小さいほど多くの特徴を選択する。
    operators=["+", "-", "*", "/", "sin", "cos", "exp", "log", "pow2", "pow3"]
)
model.fit(df)
```**`alpha`の調整**: `alpha`は試行錯誤が必要です。`LASSO selected 0 features.`と表示されたら`alpha`を小さく、特徴を選びすぎる場合は大きくしてみてください。

### 利用可能な演算子

`operators`引数に文字列のリストとして指定します。

| 演算子   | 説明              |
| :------- | :---------------- |
| `'+'`    | 加算 (a + b)      |
| `'-'`    | 減算 (a - b)      |
| `'*'`    | 乗算 (a * b)      |
| `'/'`    | 除算 (a / b)      |
| `'sin'`  | サイン (sin(a))   |
| `'cos'`  | コサイン (cos(a)) |
| `'exp'`  | 指数関数 (e^a)    |
| `'log'`  | 自然対数 (ln(a))  |
| `'sqrt'` | 平方根 (sqrt(     | a | )) ※負の値でもエラーにならない |
| `'pow2'` | 2乗 (a^2)         |
| `'pow3'` | 3乗 (a^3)         |
| `'inv'`  | 逆数 (1/a)        |

## ⚙️ APIリファレンス

### `SissoModel`

```python
class SissoModel:
    def __init__(self, n_expansion: int = 3, n_term: int = 3, k: int = 20, 
                 k_per_level: int = 200, device: str = "cpu", operators: list = None, 
                 so_method: str = "exhaustive", alpha: float = 0.01):
```

#### パラメータ
-   `n_expansion` (int, default=3): 特徴拡張の最大レベル。大きくすると探索空間が広がるが、計算時間が増加します。
-   `n_term` (int, default=3): 見つける数式モデルの最大項数。`exhaustive`サーチでのみ有効です。
-   `k` (int, default=20): SISステップで、各反復で選択する有望な特徴の数。
-   `k_per_level` (int, default=200): レベルワイズSISで、各拡張レベルで次のステップに引き継ぐ有望なレシピの数。
-   `device` (str, default="cpu"): 計算に使用するデバイス。`"cuda"`を指定するとGPU（利用可能な場合）を使用します。
-   `operators` (list[str], required): 特徴拡張に使用する演算子のリスト。
-   `so_method` (str, default="exhaustive"): モデル探索戦略。`"exhaustive"`または`"lasso"`を選択。
-   `alpha` (float, default=0.01): `so_method="lasso"`の場合に使用する正則化パラメータ。

---

### `fit(data)`

モデルを学習させます。

#### パラメータ
-   `data` (pd.DataFrame): 学習データ。**最初の列が目的変数（ターゲット）、残りの列が説明変数（特徴量）**である必要があります。

#### 戻り値
-   `self`: 学習済みの`SissoModel`インスタンス。

---

### `predict(data)`

学習済みのモデルを使って予測を行います。

#### パラメータ
-   `data` (pd.DataFrame): 予測したいデータ。**学習時と同じ列名と順序**の説明変数（特徴量）を持っている必要があります。

#### 戻り値
-   `np.ndarray`: 予測結果のNumPy配列。

---

### 学習済み属性

`fit()`の後に、以下の属性にアクセスできます。

-   `model.equation_` (str): 見つかった最良の数式モデル。
-   `model.rmse_` (float): 最良モデルの訓練データに対するRMSE。
-   `model.r2_` (float): 最良モデルの訓練データに対するR2スコア。
-   `model.best_model_recipes_` (tuple): 最良モデルを構成する`FeatureRecipe`オブジェクトのタプル。
-   `model.coeffs_` (torch.Tensor): 最良モデルの各項の係数。
-   `model.intercept_` (torch.Tensor): 最良モデルの切片。

## 📜 ライセンス

このプロジェクトはMITライセンスの下で公開されています。詳細は`LICENSE`ファイルをご覧ください。

## 🙏 謝辞

このライブラリは、オリジナルのSISSOアルゴリズムの論文に大きなインスピレーションを受けています。また、PyTorchやNumPyといった素晴らしいオープンソースプロジェクトの上に成り立っています。
