# # feature_generator.py (Level-wise SIS implemented)
# import time
# from itertools import combinations
# from typing import Dict, List, Set

# import torch

# from .executor import RecipeExecutor
# from .recipe import BinaryOperator, FeatureRecipe, Operator, UnaryOperator


# class FeatureGenerator:
#     """
#     レベルワイズSISを実装した、効率的なFeatureRecipe生成クラス。
#     """

#     def __init__(self, base_feature_names: List[str], operators: Dict[str, Operator]):
#         self.base_feature_names = base_feature_names
#         self.operators = operators
#         self.base_recipes = [FeatureRecipe(op=operators["base"], base_feature_index=i) for i in range(len(base_feature_names))]

#     def expand_with_levelwise_sis(self, n_expansion: int, k_per_level: int, executor: RecipeExecutor, y_target: torch.Tensor) -> List[FeatureRecipe]:
#         print(f"*** Starting Level-wise Recipe Generation (k_per_level={k_per_level}) ***")

#         all_promising_recipes: Set[FeatureRecipe] = set(self.base_recipes)
#         recipes_at_prev_level: Set[FeatureRecipe] = set(self.base_recipes)

#         for i in range(1, n_expansion + 1):
#             start_time = time.time()

#             # 1. 現レベルのレシピを生成
#             newly_generated_recipes = self._generate_next_level(all_promising_recipes, recipes_at_prev_level)

#             # 2. SISでスクリーニング
#             if not newly_generated_recipes:
#                 print(f"Level {i}: No new recipes generated. Stopping expansion.")
#                 break

#             promising_new_recipes = self._run_sis(list(newly_generated_recipes), executor, y_target, k_per_level)

#             # 3. 次のレベルの準備
#             all_promising_recipes.update(promising_new_recipes)
#             recipes_at_prev_level = set(promising_new_recipes)

#             print(f"Level {i}: Generated {len(newly_generated_recipes)}, selected top {len(promising_new_recipes)} promising recipes.")
#             print(f"Total promising recipes in pool: {len(all_promising_recipes)}")
#             print(f"Time for level {i}: {time.time() - start_time:.2f} seconds.")

#         return list(all_promising_recipes)

#     def _generate_next_level(self, all_recipes: Set[FeatureRecipe], prev_level_recipes: Set[FeatureRecipe]) -> Set[FeatureRecipe]:
#         next_level_recipes: Set[FeatureRecipe] = set()

#         # 二項演算子: (以前の全レシピ) x (直前のレベルのレシピ)
#         binary_ops = [op for op in self.operators.values() if isinstance(op, BinaryOperator)]
#         for op in binary_ops:
#             for r1 in all_recipes:
#                 for r2 in prev_level_recipes:
#                     if r1 != r2:
#                         next_level_recipes.add(FeatureRecipe(op=op, inputs=(r1, r2)))

#         # 単項演算子: (直前のレベルのレシピ) に適用
#         unary_ops = [op for op in self.operators.values() if isinstance(op, UnaryOperator)]
#         for op in unary_ops:
#             for r in prev_level_recipes:
#                 next_level_recipes.add(FeatureRecipe(op=op, inputs=(r,)))

#         # 既に存在するレシピを除外
#         return next_level_recipes - all_recipes

#     def _run_sis(self, recipes_to_screen: List[FeatureRecipe], executor: RecipeExecutor, target: torch.Tensor, k: int) -> List[FeatureRecipe]:
#         if not recipes_to_screen:
#             return []

#         scores = []
#         y_centered = target - target.mean()

#         for recipe in recipes_to_screen:
#             feature_tensor = executor.execute(recipe)

#             mean = torch.nanmean(feature_tensor)
#             if torch.isnan(mean):  # 全てNaNの場合
#                 scores.append(0.0)
#                 continue

#             std = torch.nanstd(feature_tensor)
#             if std == 0 or torch.isnan(std):
#                 std = 1.0

#             feature_std = (feature_tensor - mean) / std
#             feature_std = torch.nan_to_num(feature_std)

#             score = torch.abs(torch.dot(y_centered, feature_std))
#             scores.append(score.item())

#         # スコアが高い上位k個のレシピを選択
#         sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
#         top_k_indices = sorted_indices[:k]

#         return [recipes_to_screen[i] for i in top_k_indices]


# feature_generator.py (Corrected for PyTorch version compatibility)
import time
from itertools import combinations
from typing import Dict, List, Set

import torch

from .executor import RecipeExecutor
from .recipe import BinaryOperator, FeatureRecipe, Operator, UnaryOperator


class FeatureGenerator:
    """
    レベルワイズSISを実装した、効率的なFeatureRecipe生成クラス。
    """

    def __init__(self, base_feature_names: List[str], operators: Dict[str, Operator]):
        self.base_feature_names = base_feature_names
        self.operators = operators
        self.base_recipes = [FeatureRecipe(op=operators["base"], base_feature_index=i) for i in range(len(base_feature_names))]

    def expand_with_levelwise_sis(self, n_expansion: int, k_per_level: int, executor: RecipeExecutor, y_target: torch.Tensor) -> List[FeatureRecipe]:
        print(f"*** Starting Level-wise Recipe Generation (k_per_level={k_per_level}) ***")

        all_promising_recipes: Set[FeatureRecipe] = set(self.base_recipes)
        recipes_at_prev_level: Set[FeatureRecipe] = set(self.base_recipes)

        for i in range(1, n_expansion + 1):
            start_time = time.time()

            newly_generated_recipes = self._generate_next_level(all_promising_recipes, recipes_at_prev_level)

            if not newly_generated_recipes:
                print(f"Level {i}: No new recipes generated. Stopping expansion.")
                break

            promising_new_recipes = self._run_sis(list(newly_generated_recipes), executor, y_target, k_per_level)

            all_promising_recipes.update(promising_new_recipes)
            recipes_at_prev_level = set(promising_new_recipes)

            print(f"Level {i}: Generated {len(newly_generated_recipes)}, selected top {len(promising_new_recipes)} promising recipes.")
            print(f"Total promising recipes in pool: {len(all_promising_recipes)}")
            print(f"Time for level {i}: {time.time() - start_time:.2f} seconds.")

        return list(all_promising_recipes)

    def _generate_next_level(self, all_recipes: Set[FeatureRecipe], prev_level_recipes: Set[FeatureRecipe]) -> Set[FeatureRecipe]:
        next_level_recipes: Set[FeatureRecipe] = set()

        binary_ops = [op for op in self.operators.values() if isinstance(op, BinaryOperator)]
        for op in binary_ops:
            for r1 in all_recipes:
                for r2 in prev_level_recipes:
                    if r1 != r2:
                        next_level_recipes.add(FeatureRecipe(op=op, inputs=(r1, r2)))

        unary_ops = [op for op in self.operators.values() if isinstance(op, UnaryOperator)]
        for op in unary_ops:
            for r in prev_level_recipes:
                next_level_recipes.add(FeatureRecipe(op=op, inputs=(r,)))

        return next_level_recipes - all_recipes

    def _run_sis(self, recipes_to_screen: List[FeatureRecipe], executor: RecipeExecutor, target: torch.Tensor, k: int) -> List[FeatureRecipe]:
        if not recipes_to_screen:
            return []

        scores = []
        # y_targetもNaNを含む可能性があるので、安全な操作を行う
        valid_y_mask = ~torch.isnan(target)
        y_centered = target[valid_y_mask] - target[valid_y_mask].mean()

        for recipe in recipes_to_screen:
            feature_tensor = executor.execute(recipe)

            # 共通の有効なインデックスを取得
            valid_mask = ~torch.isnan(feature_tensor) & valid_y_mask

            valid_feature = feature_tensor[valid_mask]

            if valid_feature.numel() < 2:  # 有効なデータが2点未満の場合
                scores.append(0.0)
                continue

            # ===== 修正箇所 =====
            mean = valid_feature.mean()
            std = valid_feature.std()
            # ====================

            if std > 0:
                feature_std = (valid_feature - mean) / std
                # ターゲットも同じマスクでフィルタリング
                y_filtered = target[valid_mask] - target[valid_mask].mean()
                score = torch.abs(torch.dot(y_filtered, feature_std))
                scores.append(score.item())
            else:
                scores.append(0.0)

        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        top_k_indices = sorted_indices[:k]

        return [recipes_to_screen[i] for i in top_k_indices]
