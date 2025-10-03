# import time
# from typing import List, Tuple

# import numpy as np

# # model.py (Final and Corrected Version)
# import pandas as pd
# import torch

# from .executor import RecipeExecutor
# from .feature_generator import FeatureGenerator
# from .recipe import OPERATORS, FeatureRecipe
# from .regressor import SissoRegressor


# class SissoModel:
#     def __init__(self, n_expansion: int = 3, n_term: int = 3, k: int = 20, k_per_level: int = 200, device: str = "cpu", operators: list = None, so_method: str = "exhaustive", alpha: float = 0.01):
#         self.n_expansion = n_expansion
#         self.n_term = n_term
#         self.k = k
#         self.k_per_level = k_per_level
#         self.so_method = so_method
#         self.alpha = alpha
#         self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
#         self.operators = {op: OPERATORS[op] for op in operators + ["base"] if op in OPERATORS} if operators else OPERATORS

#         self.best_model_recipes_: Tuple[FeatureRecipe, ...] = None
#         self.coeffs_: torch.Tensor = None
#         self.intercept_: torch.Tensor = None
#         self.base_feature_names_: List[str] = None
#         self.equation_: str = ""
#         self.rmse_: float = float("inf")
#         self.r2_: float = -float("inf")

#         print(f"Model initialized on device: {self.device}")

#     def fit(self, data: pd.DataFrame):
#         start_time = time.time()
#         df = data.copy().select_dtypes(include=np.number)
#         y = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).to(self.device)
#         X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32).to(self.device)
#         self.base_feature_names_ = df.columns[1:].tolist()

#         executor = RecipeExecutor(X)
#         generator = FeatureGenerator(self.base_feature_names_, self.operators)
#         recipes = generator.expand_with_levelwise_sis(self.n_expansion, self.k_per_level, executor, y)

#         regressor = SissoRegressor(recipes, executor, y, self.n_term, self.k, self.so_method, self.alpha)
#         result = regressor.fit()

#         print(f"\n{'='*50}\nSISSO fitting finished. Total time: {time.time() - start_time:.2f}s\n{'='*50}")

#         if result:
#             rmse, final_equation, r2, all_models = result
#             if not all_models:
#                 print("\nCould not find a valid model.")
#                 return self

#             # ★★★ CRITICAL CORRECTION ★★★
#             # Find the best model from the returned dictionary
#             best_model = min(all_models.values(), key=lambda m: m["rmse"])

#             # Store all the properties from the best model
#             self.best_model_recipes_ = best_model["recipes"]
#             self.coeffs_ = best_model["coeffs"]
#             self.intercept_ = best_model["intercept"]
#             self.rmse_ = best_model["rmse"]

#             # Use the final_equation string that was already correctly formatted by the regressor
#             self.equation_ = final_equation
#             self.r2_ = r2
#             # ★★★★★★★★★★★★★★★★★★★★★★★★★★

#             print(f"\nBest Model Found ({len(self.best_model_recipes_)} terms):\n  RMSE: {self.rmse_:.6f}\n  R2:   {self.r2_:.6f}\n  Equation: {self.equation_}")
#         else:
#             print("\nCould not find a valid model.")
#         return self

#     def predict(self, data: pd.DataFrame) -> np.ndarray:
#         if self.best_model_recipes_ is None:
#             raise RuntimeError("Model not fitted or no valid model was found.")
#         if list(data.columns) != self.base_feature_names_:
#             raise ValueError("Input columns do not match training columns.")
#         X_test = torch.tensor(data.values, dtype=torch.float32).to(self.device)
#         pred_executor = RecipeExecutor(X_test)

#         y_pred = torch.full((X_test.shape[0],), self.intercept_.item(), device=self.device)
#         for i, recipe in enumerate(self.best_model_recipes_):
#             y_pred += self.coeffs_.flatten()[i] * torch.nan_to_num(pred_executor.execute(recipe))
#         return y_pred.cpu().detach().numpy()

import time
from typing import List, Tuple

import numpy as np

# model.py (Add one line to set feature names)
import pandas as pd
import torch

from .executor import RecipeExecutor
from .feature_generator import FeatureGenerator
from .recipe import OPERATORS, FeatureRecipe  # FeatureRecipeをインポート
from .regressor import SissoRegressor


class SissoModel:
    # ... (__init__は変更なし) ...
    def __init__(self, n_expansion: int = 3, n_term: int = 3, k: int = 20, k_per_level: int = 200, device: str = "cpu", operators: list = None, so_method: str = "exhaustive", alpha: float = 0.01):
        self.n_expansion = n_expansion
        self.n_term = n_term
        self.k = k
        self.k_per_level = k_per_level
        self.so_method = so_method
        self.alpha = alpha
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.operators = {op: OPERATORS[op] for op in operators + ["base"] if op in OPERATORS} if operators else OPERATORS

        self.best_model_recipes_: Tuple[FeatureRecipe, ...] = None
        self.coeffs_: torch.Tensor = None
        self.intercept_: torch.Tensor = None
        self.base_feature_names_: List[str] = None
        self.equation_ = ""
        self.rmse_ = float("inf")
        self.r2_ = -float("inf")

        print(f"Model initialized on device: {self.device}")

    def fit(self, data: pd.DataFrame):
        start_time = time.time()
        df = data.copy().select_dtypes(include=np.number)
        y = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).to(self.device)
        X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32).to(self.device)
        self.base_feature_names_ = df.columns[1:].tolist()

        # ★★★ CRITICAL ADDITION ★★★
        # 読み込んだ特徴量名をFeatureRecipeクラスに設定する
        FeatureRecipe.base_feature_names = self.base_feature_names_
        # ★★★★★★★★★★★★★★★★★★★★★★★

        executor = RecipeExecutor(X)
        generator = FeatureGenerator(self.base_feature_names_, self.operators)
        recipes = generator.expand_with_levelwise_sis(self.n_expansion, self.k_per_level, executor, y)

        regressor = SissoRegressor(recipes, executor, y, self.n_term, self.k, self.so_method, self.alpha)
        result = regressor.fit()

        print(f"\n{'='*50}\nSISSO fitting finished. Total time: {time.time() - start_time:.2f}s\n{'='*50}")

        if result:
            rmse, eq, r2, all_models = result
            if not all_models:
                print("\nCould not find a valid model.")
                return self
            best_model = min(all_models.values(), key=lambda m: m["rmse"])
            self.best_model_recipes_ = best_model["recipes"]
            self.coeffs_ = best_model["coeffs"]
            self.intercept_ = best_model["intercept"]
            self.equation_ = eq
            self.rmse_ = best_model["rmse"]
            self.r2_ = r2

            print(f"\nBest Model Found ({len(self.best_model_recipes_)} terms):\n  RMSE: {self.rmse_:.6f}\n  R2:   {self.r2_:.6f}\n  Equation: {self.equation_}")
        else:
            print("\nCould not find a valid model.")

        # クリーンアップ（次のfit呼び出しに影響を与えないように）
        FeatureRecipe.base_feature_names = []
        return self

    # ... (predictメソッドは変更なし) ...
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.best_model_recipes_ is None:
            raise RuntimeError("Model not fitted or no valid model was found.")
        if list(data.columns) != self.base_feature_names_:
            raise ValueError("Input columns do not match training columns.")

        # ★★★ PREDICT CORRECTION ★★★
        # predict時にも特徴量名を設定
        FeatureRecipe.base_feature_names = self.base_feature_names_

        X_test = torch.tensor(data.values, dtype=torch.float32).to(self.device)
        pred_executor = RecipeExecutor(X_test)

        y_pred = torch.full((X_test.shape[0],), self.intercept_.item(), device=self.device)
        for i, recipe in enumerate(self.best_model_recipes_):
            y_pred += self.coeffs_.flatten()[i] * torch.nan_to_num(pred_executor.execute(recipe))

        # クリーンアップ
        FeatureRecipe.base_feature_names = []
        return y_pred.cpu().detach().numpy()
