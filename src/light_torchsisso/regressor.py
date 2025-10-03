import time
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np

# regressor.py (Lasso support added)
import torch
from sklearn.linear_model import Lasso

from .executor import RecipeExecutor
from .recipe import FeatureRecipe


class SissoRegressor:
    def __init__(self, all_recipes: List[FeatureRecipe], executor: RecipeExecutor, y: torch.Tensor, n_term: int, k: int, so_method: str = "exhaustive", alpha: float = 0.01):
        self.all_recipes = all_recipes
        self.executor = executor
        self.y = y
        self.n_term = n_term
        self.k = k
        self.so_method = so_method
        self.alpha = alpha  # alpha for Lasso
        self.device = executor.device

        self.y_mean = y.mean()
        self.y_centered = y - self.y_mean

        self.best_models: Dict[int, dict] = {}

    def _format_equation(self, recipes: Tuple[FeatureRecipe, ...], coeffs: torch.Tensor, intercept: torch.Tensor) -> str:
        equation = "".join(f"{c.item():+.6f} * {repr(r)} " for r, c in zip(recipes, coeffs.flatten()))
        return equation + f"{intercept.item():+.6f}"

    def _run_sis(self, target: torch.Tensor, recipes: List[FeatureRecipe]) -> List[FeatureRecipe]:
        if not recipes:
            return []
        scores = []
        for recipe in recipes:
            tensor = self.executor.execute(recipe)
            valid = ~torch.isnan(tensor) & ~torch.isnan(target)
            if valid.sum() < 2:
                scores.append(0.0)
                continue
            valid_f, valid_t = tensor[valid], target[valid]
            mean, std = valid_f.mean(), valid_f.std()
            if std > 1e-8:
                scores.append(torch.abs(torch.dot(valid_t - valid_t.mean(), (valid_f - mean) / std)).item())
            else:
                scores.append(0.0)
        return [recipes[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[: self.k]]

    def _get_final_model(self, model_recipes_list: list) -> Tuple:
        """Given a list of recipes, computes the OLS model and returns its properties."""
        num_terms = len(model_recipes_list)
        n_samples = self.y.shape[0]
        y_np = self.y_centered.cpu().numpy()

        X = torch.stack([self.executor.execute(r) for r in model_recipes_list], dim=1)
        X_np = X.cpu().numpy()

        if np.isnan(X_np).any():
            col_mean = np.nanmean(X_np, axis=0)
            X_np[np.where(np.isnan(X_np))] = np.take(col_mean, np.where(np.isnan(X_np))[1])

        X_mean, X_std = np.mean(X_np, axis=0), np.std(X_np, axis=0)
        X_std[X_std < 1e-8] = 1.0
        X_std_np = (X_np - X_mean) / X_std

        try:
            coeffs_std, residuals, _, _ = np.linalg.lstsq(X_std_np, y_np, rcond=None)
            res_val = residuals[0] if isinstance(residuals, np.ndarray) and len(residuals) > 0 else float("inf")
            rmse = np.sqrt(res_val / n_samples)

            coeffs = torch.tensor(coeffs_std / X_std, device=self.device, dtype=torch.float32)
            intercept = self.y_mean - torch.dot(coeffs, torch.tensor(X_mean, device=self.device, dtype=torch.float32))

            return rmse, tuple(model_recipes_list), coeffs, intercept
        except np.linalg.LinAlgError:
            return float("inf"), None, None, None

    def _run_so_lasso(self):
        print(f"--- Running SO with LASSO (alpha={self.alpha}). Candidates: {len(self.all_recipes)} ---")

        X_candidates, valid_recipes = [], []
        for recipe in self.all_recipes:
            tensor = self.executor.execute(recipe)
            valid = ~torch.isnan(tensor)
            if valid.sum() > 1:
                valid_f = tensor[valid]
                mean, std = valid_f.mean(), valid_f.std()
                if std > 1e-8:
                    std_tensor = torch.zeros_like(tensor)
                    std_tensor[valid] = (valid_f - mean) / std
                    X_candidates.append(std_tensor)
                    valid_recipes.append(recipe)

        if not X_candidates:
            print("No valid features for LASSO.")
            return

        X_matrix = torch.stack(X_candidates, dim=1).cpu().numpy()
        lasso = Lasso(alpha=self.alpha, max_iter=3000, random_state=42, tol=1e-4)
        lasso.fit(X_matrix, self.y_centered.cpu().numpy())

        selected_indices = np.where(np.abs(lasso.coef_) > 1e-6)[0]
        if len(selected_indices) == 0:
            print("LASSO selected 0 features.")
            return

        model_recipes = [valid_recipes[i] for i in selected_indices]
        num_terms = len(model_recipes)
        print(f"LASSO selected {num_terms} features.")

        rmse, recipes_tuple, coeffs, intercept = self._get_final_model(model_recipes)
        if recipes_tuple:
            self.best_models[num_terms] = {"rmse": rmse, "recipes": recipes_tuple, "coeffs": coeffs, "intercept": intercept}
            print(f"Found a {num_terms}-term model via LASSO. RMSE: {rmse:.6f}")
            print(f"Equation: {self._format_equation(recipes_tuple, coeffs, intercept)}")

    def fit(self):
        print(f"***************** Starting SISSO Regressor (Method: {self.so_method}) *****************")

        if self.so_method == "lasso":
            self._run_so_lasso()

        elif self.so_method == "exhaustive":
            residual, pool = self.y_centered, []
            for i in range(1, self.n_term + 1):
                start_time = time.time()
                print(f"\n===== Searching for {i}-term models =====")

                top_k = self._run_sis(residual, [r for r in self.all_recipes if r not in pool])
                pool.extend(top_k)
                print(f"SIS selected {len(top_k)} new features. Pool size: {len(pool)}")

                combos = list(combinations(pool, i))
                if not combos:
                    continue
                print(f"--- Running SO for {i}-term models. Total combinations: {len(combos)} ---")

                best_rmse, best_model = float("inf"), None
                for combo in combos:
                    rmse, recipes, coeffs, intercept = self._get_final_model(list(combo))
                    if rmse < best_rmse:
                        best_rmse, best_model = rmse, {"r": recipes, "c": coeffs, "i": intercept}

                if best_model:
                    self.best_models[i] = {"rmse": best_rmse, "recipes": best_model["r"], "coeffs": best_model["c"], "intercept": best_model["i"]}
                    y_pred = best_model["i"] + sum(c * torch.nan_to_num(self.executor.execute(r)) for c, r in zip(best_model["c"], best_model["r"]))
                    residual = self.y - y_pred
                    print(f"Best {i}-term model: RMSE={best_rmse:.6f}, Eq: {self._format_equation(best_model['r'], best_model['c'], best_model['i'])}")
                else:
                    print(f"No valid model found for term {i}.")
                print(f"Time: {time.time() - start_time:.2f} seconds")

        if not self.best_models:
            print("Fit finished, but no valid models were found.")
            return None

        best_term, best_model = min(self.best_models.items(), key=lambda item: item[1]["rmse"])
        r2 = 1.0 - (best_model["rmse"] ** 2 * self.y.shape[0]) / torch.sum((self.y_centered[~torch.isnan(self.y_centered)]) ** 2).item()
        final_equation = self._format_equation(best_model["recipes"], best_model["coeffs"], best_model["intercept"])
        return best_model["rmse"], final_equation, r2, self.best_models
