# import time
# from typing import Dict, List, Tuple

# import numpy as np

# # model.py (Corrected for Lasso logic and returning self)
# import pandas as pd
# import torch

# from .executor import RecipeExecutor
# from .feature_generator import FeatureGenerator
# from .recipe import OPERATORS, FeatureRecipe
# from .regressor import SissoRegressor


# class SissoModel:
#     def __init__(self, n_expansion: int = 3, n_term: int = 3, k: int = 20, so_method: str = "exhaustive", k_per_level: int = 200, device: str = "cpu", operators: list = None):
#         self.n_expansion = n_expansion
#         self.n_term = n_term
#         self.k = k
#         self.k_per_level = k_per_level
#         self.so_method = so_method
#         self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

#         if operators is None:
#             self.operators = OPERATORS
#             print("Warning: No operator set provided. Using default set.")
#         else:
#             self.operators = {op_name: OPERATORS[op_name] for op_name in operators + ["base"] if op_name in OPERATORS}

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

#         df = data.copy().select_dtypes(include=["float64", "int64", "float32", "int32"])
#         y = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).to(self.device)
#         X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32).to(self.device)
#         self.base_feature_names_ = df.columns[1:].tolist()

#         executor = RecipeExecutor(X)

#         generator = FeatureGenerator(self.base_feature_names_, self.operators)
#         promising_recipes = generator.expand_with_levelwise_sis(self.n_expansion, self.k_per_level, executor, y)

#         regressor = SissoRegressor(promising_recipes, executor, y, self.n_term, self.k, self.so_method)
#         result = regressor.fit()

#         print("\n" + "=" * 50 + f"\nSISSO fitting process finished. Total time: {time.time() - start_time:.2f}s\n" + "=" * 50)

#         if result:
#             rmse, equation, r2, all_models = result
#             if not all_models:
#                 print("\nCould not find a valid model.")
#                 return self

#             best_term, best_model = min(all_models.items(), key=lambda item: item[1]["rmse"])

#             self.best_model_recipes_ = best_model["recipes"]
#             self.coeffs_ = best_model["coeffs"]
#             self.intercept_ = best_model["intercept"]
#             self.equation_ = equation
#             self.rmse_ = rmse
#             self.r2_ = r2

#             print(f"\nBest Model Found ({len(self.best_model_recipes_)} terms):")
#             print(f"  RMSE: {self.rmse_:.6f}")
#             print(f"  R2:   {self.r2_:.6f}")
#             print(f"  Equation: {self.equation_}")
#         else:
#             print("\nCould not find a valid model.")

#         return self

#     def predict(self, data: pd.DataFrame) -> np.ndarray:
#         if self.best_model_recipes_ is None:
#             raise RuntimeError("Model has not been fitted yet, or no valid model was found. Please call fit() first.")

#         if list(data.columns) != self.base_feature_names_:
#             raise ValueError(f"Input columns {list(data.columns)} do not match training columns {self.base_feature_names_}")

#         X_test = torch.tensor(data.values, dtype=torch.float32).to(self.device)

#         pred_executor = RecipeExecutor(X_test)

#         y_pred = self.intercept_.clone()
#         coeffs_flat = self.coeffs_.flatten()
#         for i, recipe in enumerate(self.best_model_recipes_):
#             feature_tensor = pred_executor.execute(recipe)
#             y_pred += coeffs_flat[i] * torch.nan_to_num(feature_tensor)

#         return y_pred.cpu().detach().numpy()


# import time
# from typing import List, Tuple

# import numpy as np

# # model.py (Final version with predict fix and debug logs)
# import pandas as pd
# import torch

# from .executor import RecipeExecutor
# from .feature_generator import FeatureGenerator
# from .recipe import OPERATORS, FeatureRecipe
# from .regressor import SissoRegressor


# class SissoModel:
#     def __init__(self, n_expansion: int = 3, n_term: int = 3, k: int = 20, k_per_level: int = 200, device: str = "cpu", operators: list = None, so_method: str = "exhaustive"):
#         self.n_expansion = n_expansion
#         self.n_term = n_term
#         self.k = k
#         self.k_per_level = k_per_level
#         self.so_method = so_method
#         self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
#         self.operators = {op: OPERATORS[op] for op in operators + ["base"] if op in OPERATORS} if operators else OPERATORS

#         self.best_model_recipes_: Tuple[FeatureRecipe, ...] = None
#         self.coeffs_: torch.Tensor = None
#         self.intercept_: torch.Tensor = None
#         self.base_feature_names_: List[str] = None
#         self.equation_ = ""
#         self.rmse_ = float("inf")
#         self.r2_ = -float("inf")

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
#         regressor = SissoRegressor(recipes, executor, y, self.n_term, self.k, self.so_method)
#         result = regressor.fit()

#         print(f"\n{'='*50}\nSISSO fitting finished. Total time: {time.time() - start_time:.2f}s\n{'='*50}")

#         if result:
#             rmse, eq, r2, all_models = result
#             if not all_models:
#                 print("\nCould not find a valid model.")
#                 return self
#             best_model = min(all_models.values(), key=lambda m: m["rmse"])
#             self.best_model_recipes_ = best_model["recipes"]
#             self.coeffs_ = best_model["coeffs"]
#             self.intercept_ = best_model["intercept"]
#             self.equation_ = eq  # Use the final equation string from the result
#             self.rmse_ = best_model["rmse"]  # Use the specific rmse of the best model
#             self.r2_ = r2  # R2 is calculated based on the best model, so it's correct

#             print(f"\nBest Model Found ({len(self.best_model_recipes_)} terms):\n  RMSE: {self.rmse_:.6f}\n  R2:   {self.r2_:.6f}\n  Equation: {self.equation_}")
#         else:
#             print("\nCould not find a valid model.")
#         return self

#     def predict(self, data: pd.DataFrame, debug: bool = False) -> np.ndarray:
#         """
#         学習済みモデルを使用して新しいデータの予測を行う。
#         debug=Trueにすると、計算過程の形状情報を表示する。
#         """
#         if self.best_model_recipes_ is None:
#             raise RuntimeError("Model has not been fitted yet, or no valid model was found.")
#         if list(data.columns) != self.base_feature_names_:
#             raise ValueError("Input columns do not match training columns.")

#         X_test = torch.tensor(data.values, dtype=torch.float32).to(self.device)
#         pred_executor = RecipeExecutor(X_test)

#         if debug:
#             print("\n--- Starting Predict Debug ---")

#         # ★★★ FINAL CORRECTION ★★★
#         # y_predの初期化を、入力データと同じサンプル数のテンソルで行う
#         y_pred = torch.full((X_test.shape[0],), self.intercept_.item(), device=self.device)
#         if debug:
#             print(f"  - Initial y_pred shape (from intercept): {y_pred.shape}")
#         # ★★★★★★★★★★★★★★★★★★★★

#         coeffs_flat = self.coeffs_.flatten()
#         for i, recipe in enumerate(self.best_model_recipes_):
#             # 1. レシピから特徴量テンソルを計算
#             feature_tensor = pred_executor.execute(recipe)
#             feature_tensor_clean = torch.nan_to_num(feature_tensor)
#             if debug:
#                 print(f"\n  - Term {i+1}: Recipe = {repr(recipe)}")
#             if debug:
#                 print(f"    - Feature tensor shape: {feature_tensor_clean.shape}")

#             # 2. 係数を取得
#             coeff = coeffs_flat[i]
#             if debug:
#                 print(f"    - Coefficient value: {coeff.item():.4f}, shape: {coeff.shape}")

#             # 3. 項を計算
#             term = coeff * feature_tensor_clean
#             if debug:
#                 print(f"    - Calculated term shape: {term.shape}")

#             # 4. y_predに加算
#             y_pred += term
#             if debug:
#                 print(f"    - y_pred shape after addition: {y_pred.shape}")

#         if debug:
#             print("--- Predict Debug Finished ---\n")

#         return y_pred.cpu().detach().numpy()

import time
from typing import List, Tuple

import numpy as np

# model.py (Final and Corrected Version)
import pandas as pd
import torch

from .executor import RecipeExecutor
from .feature_generator import FeatureGenerator
from .recipe import OPERATORS, FeatureRecipe
from .regressor import SissoRegressor


class SissoModel:
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
        self.equation_: str = ""
        self.rmse_: float = float("inf")
        self.r2_: float = -float("inf")

        print(f"Model initialized on device: {self.device}")

    def fit(self, data: pd.DataFrame):
        start_time = time.time()
        df = data.copy().select_dtypes(include=np.number)
        y = torch.tensor(df.iloc[:, 0].values, dtype=torch.float32).to(self.device)
        X = torch.tensor(df.iloc[:, 1:].values, dtype=torch.float32).to(self.device)
        self.base_feature_names_ = df.columns[1:].tolist()

        executor = RecipeExecutor(X)
        generator = FeatureGenerator(self.base_feature_names_, self.operators)
        recipes = generator.expand_with_levelwise_sis(self.n_expansion, self.k_per_level, executor, y)

        regressor = SissoRegressor(recipes, executor, y, self.n_term, self.k, self.so_method, self.alpha)
        result = regressor.fit()

        print(f"\n{'='*50}\nSISSO fitting finished. Total time: {time.time() - start_time:.2f}s\n{'='*50}")

        if result:
            rmse, final_equation, r2, all_models = result
            if not all_models:
                print("\nCould not find a valid model.")
                return self

            # ★★★ CRITICAL CORRECTION ★★★
            # Find the best model from the returned dictionary
            best_model = min(all_models.values(), key=lambda m: m["rmse"])

            # Store all the properties from the best model
            self.best_model_recipes_ = best_model["recipes"]
            self.coeffs_ = best_model["coeffs"]
            self.intercept_ = best_model["intercept"]
            self.rmse_ = best_model["rmse"]

            # Use the final_equation string that was already correctly formatted by the regressor
            self.equation_ = final_equation
            self.r2_ = r2
            # ★★★★★★★★★★★★★★★★★★★★★★★★★★

            print(f"\nBest Model Found ({len(self.best_model_recipes_)} terms):\n  RMSE: {self.rmse_:.6f}\n  R2:   {self.r2_:.6f}\n  Equation: {self.equation_}")
        else:
            print("\nCould not find a valid model.")
        return self

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.best_model_recipes_ is None:
            raise RuntimeError("Model not fitted or no valid model was found.")
        if list(data.columns) != self.base_feature_names_:
            raise ValueError("Input columns do not match training columns.")
        X_test = torch.tensor(data.values, dtype=torch.float32).to(self.device)
        pred_executor = RecipeExecutor(X_test)

        y_pred = torch.full((X_test.shape[0],), self.intercept_.item(), device=self.device)
        for i, recipe in enumerate(self.best_model_recipes_):
            y_pred += self.coeffs_.flatten()[i] * torch.nan_to_num(pred_executor.execute(recipe))
        return y_pred.cpu().detach().numpy()
