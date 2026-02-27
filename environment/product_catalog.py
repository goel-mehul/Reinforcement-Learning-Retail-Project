import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Product:
    """Represents a single product in the retail environment."""
    product_id: int
    name: str
    category: str
    base_cost: float          # what retailers pay suppliers
    base_retail_price: float  # suggested retail price
    promo_eligible: bool      # can this product be put on promotion
    seasonality: np.ndarray   # 365-day demand multiplier curve
    brand_strength: float     # 0-1, how strong brand loyalty is for this product

    def margin(self, price: float) -> float:
        """Profit margin at a given price."""
        return (price - self.base_cost) / price

    def is_priced_below_cost(self, price: float) -> bool:
        return price < self.base_cost


class ProductCatalog:
    """
    Generates and manages all products in the retail simulation.
    Calibrated from Instacart dataset EDA.
    """

    # Category config: (n_products, price_range, cost_pct_range, promo_rate, brand_strength_range)
    CATEGORY_CONFIG = {
        "produce":    (12, (0.99,  8.99),  (0.45, 0.60), 0.70, (0.20, 0.45)),
        "dairy_eggs": (12, (1.49, 12.99),  (0.50, 0.65), 0.60, (0.40, 0.65)),
        "snacks":     (10, (1.99, 11.99),  (0.35, 0.55), 0.80, (0.50, 0.80)),
        "beverages":  (10, (0.99, 14.99),  (0.35, 0.60), 0.75, (0.45, 0.75)),
        "frozen":     (6,  (2.99, 15.99),  (0.40, 0.60), 0.65, (0.35, 0.60)),
    }

    def __init__(self, config_path: Optional[Path] = None, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.config = self._load_config(config_path)
        self.products: Dict[int, Product] = {}
        self._generate_catalog()

    def _load_config(self, config_path: Optional[Path]) -> dict:
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "env_config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)

    def _generate_catalog(self):
        """Generate all products across all categories."""
        product_id = 0
        for category, (n_products, price_range, cost_pct_range,
                        promo_rate, brand_range) in self.CATEGORY_CONFIG.items():
            for i in range(n_products):
                base_retail = round(
                    self.rng.uniform(*price_range), 2
                )
                cost_pct = self.rng.uniform(*cost_pct_range)
                base_cost = round(base_retail * cost_pct, 2)
                brand_strength = round(self.rng.uniform(*brand_range), 3)
                promo_eligible = self.rng.random() < promo_rate

                seasonality = self._generate_seasonality(category)

                product = Product(
                    product_id=product_id,
                    name=f"{category}_{i:02d}",
                    category=category,
                    base_cost=base_cost,
                    base_retail_price=base_retail,
                    promo_eligible=promo_eligible,
                    seasonality=seasonality,
                    brand_strength=brand_strength,
                )
                self.products[product_id] = product
                product_id += 1

    def _generate_seasonality(self, category: str) -> np.ndarray:
        """
        Generate a 365-day seasonality multiplier curve for a product.
        Different categories have different seasonal patterns.
        Calibrated from Instacart EDA — produce peaks in summer,
        frozen peaks in winter, beverages peak in summer, etc.
        """
        days = np.arange(365)
        base = np.ones(365)

        if category == "produce":
            # peaks in summer (day ~180)
            seasonal = 0.25 * np.sin(2 * np.pi * (days - 60) / 365)

        elif category == "frozen":
            # peaks in winter (day ~0 and ~365)
            seasonal = 0.20 * np.cos(2 * np.pi * days / 365)

        elif category == "beverages":
            # peaks in summer, slight dip in winter
            seasonal = 0.30 * np.sin(2 * np.pi * (days - 80) / 365)

        elif category == "dairy_eggs":
            # relatively flat with slight holiday bump
            seasonal = 0.10 * np.sin(2 * np.pi * (days - 30) / 365)

        else:  # snacks
            # peaks around holidays (end of year)
            seasonal = 0.15 * np.cos(2 * np.pi * (days - 180) / 365)

        # add small random noise per product for variety
        noise = self.rng.normal(0, 0.02, 365)
        curve = base + seasonal + noise
        return np.clip(curve, 0.5, 2.5)  # keep multipliers in sane range

    # ── Public API ────────────────────────────────────────────────

    def get_product(self, product_id: int) -> Product:
        return self.products[product_id]

    def get_category(self, category: str) -> List[Product]:
        return [p for p in self.products.values() if p.category == category]

    def get_all_products(self) -> List[Product]:
        return list(self.products.values())

    def get_promo_eligible(self) -> List[Product]:
        return [p for p in self.products.values() if p.promo_eligible]

    def summary(self) -> pd.DataFrame:
        """Returns a DataFrame summary of the full catalog."""
        rows = []
        for p in self.products.values():
            rows.append({
                "product_id":        p.product_id,
                "name":              p.name,
                "category":          p.category,
                "base_cost":         p.base_cost,
                "base_retail_price": p.base_retail_price,
                "margin_pct":        round(p.margin(p.base_retail_price) * 100, 1),
                "brand_strength":    p.brand_strength,
                "promo_eligible":    p.promo_eligible,
            })
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self.products)

    def __repr__(self) -> str:
        return f"ProductCatalog({len(self)} products across {len(self.CATEGORY_CONFIG)} categories)"