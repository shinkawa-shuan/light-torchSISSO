# # recipe.py
# from typing import Callable, List, Tuple

# import torch


# class FeatureRecipe:
#     """
#     特徴量を計算するための軽量な「設計図」。
#     実際のテンソルデータは保持せず、計算方法のみを定義する。
#     """

#     def __init__(self, op: "Operator", inputs: Tuple["FeatureRecipe", ...] = (), base_feature_index: int = -1):
#         self.op = op
#         self.inputs = inputs
#         self.base_feature_index = base_feature_index
#         # 演算の可換性（a+b = b+a）を考慮してハッシュ値を計算
#         if op.is_commutative and len(inputs) > 1:
#             self._hash = hash((self.op.name, tuple(sorted(self.inputs, key=hash))))
#         else:
#             self._hash = hash((self.op.name, self.inputs, self.base_feature_index))

#     def __hash__(self):
#         return self._hash

#     def __eq__(self, other):
#         return isinstance(other, FeatureRecipe) and self._hash == other._hash

#     def __repr__(self):
#         # デバッグ用に人間が読める形式の数式文字列を生成
#         return self.op.format_string(self.inputs, self.base_feature_index)


# class Operator:
#     """
#     全ての演算子の基底クラス。
#     """

#     def __init__(self, name: str, torch_func: Callable, is_commutative: bool = False):
#         self.name = name
#         self.torch_func = torch_func
#         self.is_commutative = is_commutative

#     def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int) -> str:
#         raise NotImplementedError


# class UnaryOperator(Operator):
#     """単項演算子 (例: exp(x), sqrt(x))"""

#     def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int) -> str:
#         return f"{self.name}({repr(inputs[0])})"


# class BinaryOperator(Operator):
#     """二項演算子 (例: x+y, x*y)"""

#     def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int) -> str:
#         return f"({repr(inputs[0])} {self.name} {repr(inputs[1])})"


# class BaseFeatureOperator(Operator):
#     """ベース特徴量そのものを表す特殊な演算子"""

#     def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int) -> str:
#         return f"f{base_feature_index}"


# # --- 利用可能な演算子の定義 ---
# OPERATORS = {
#     # Base
#     "base": BaseFeatureOperator("base", lambda x: x),
#     # Binary
#     "+": BinaryOperator("+", torch.add, is_commutative=True),
#     "-": BinaryOperator("-", torch.sub),
#     "*": BinaryOperator("*", torch.mul, is_commutative=True),
#     "/": BinaryOperator("/", torch.div),
#     # Unary
#     "exp": UnaryOperator("exp", torch.exp),
#     "log": UnaryOperator("log", torch.log),
#     "sin": UnaryOperator("sin", torch.sin),
#     "cos": UnaryOperator("cos", torch.cos),
#     "sqrt": UnaryOperator("sqrt", lambda x: torch.sqrt(torch.abs(x))),
#     "pow2": UnaryOperator("^2", lambda x: torch.pow(x, 2)),
#     "pow3": UnaryOperator("^3", lambda x: torch.pow(x, 3)),
#     "inv": UnaryOperator("^-1", torch.reciprocal),
# }

# recipe.py (Final version with original feature name support)
from typing import Callable, List, Tuple

import torch


class FeatureRecipe:
    """
    特徴量を計算するための軽量な「設計図」。
    """

    # ★★★ MODIFICATION ★★★
    # クラス変数として特徴量名のリストを保持
    base_feature_names: List[str] = []

    def __init__(self, op: "Operator", inputs: Tuple["FeatureRecipe", ...] = (), base_feature_index: int = -1):
        self.op = op
        self.inputs = inputs
        self.base_feature_index = base_feature_index
        if op.is_commutative and len(inputs) > 1:
            self._hash = hash((self.op.name, tuple(sorted(self.inputs, key=hash))))
        else:
            self._hash = hash((self.op.name, self.inputs, self.base_feature_index))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return isinstance(other, FeatureRecipe) and self._hash == other._hash

    def __repr__(self):
        # ★★★ MODIFICATION ★★★
        # ここで base_feature_names を使って表示を切り替える
        return self.op.format_string(self.inputs, self.base_feature_index, self.base_feature_names)


class Operator:
    """
    全ての演算子の基底クラス。
    """

    def __init__(self, name: str, torch_func: Callable, is_commutative: bool = False):
        self.name, self.torch_func, self.is_commutative = name, torch_func, is_commutative

    # ★★★ MODIFICATION ★★★
    # format_stringにfeature_names引数を追加
    def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int, feature_names: List[str]) -> str:
        raise NotImplementedError


class UnaryOperator(Operator):
    def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int, feature_names: List[str]) -> str:
        return f"{self.name}({repr(inputs[0])})"


class BinaryOperator(Operator):
    def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int, feature_names: List[str]) -> str:
        return f"({repr(inputs[0])} {self.name} {repr(inputs[1])})"


class BaseFeatureOperator(Operator):
    def format_string(self, inputs: Tuple[FeatureRecipe, ...], base_feature_index: int, feature_names: List[str]) -> str:
        # ★★★ MODIFICATION ★★★
        # インデックスが有効範囲内なら、特徴量名リストから名前を返す
        if feature_names and 0 <= base_feature_index < len(feature_names):
            return feature_names[base_feature_index]
        # リストがない、またはインデックスが無効な場合は、従来のf-string形式に戻る
        return f"f{base_feature_index}"


# (OPERATORSの定義は変更なし)
OPERATORS = {
    "base": BaseFeatureOperator("base", lambda x: x),
    "+": BinaryOperator("+", torch.add, is_commutative=True),
    "-": BinaryOperator("-", torch.sub),
    "*": BinaryOperator("*", torch.mul, is_commutative=True),
    "/": BinaryOperator("/", torch.div),
    "exp": UnaryOperator("exp", torch.exp),
    "log": UnaryOperator("log", torch.log),
    "sin": UnaryOperator("sin", torch.sin),
    "cos": UnaryOperator("cos", torch.cos),
    "sqrt": UnaryOperator("sqrt", lambda x: torch.sqrt(torch.abs(x))),
    "pow2": UnaryOperator("^2", lambda x: torch.pow(x, 2)),
    "pow3": UnaryOperator("^3", lambda x: torch.pow(x, 3)),
    "inv": UnaryOperator("^-1", torch.reciprocal),
}
