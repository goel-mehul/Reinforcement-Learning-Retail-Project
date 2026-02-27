import numpy as np
import pytest
from environment.product_catalog import Product, ProductCatalog


class TestProduct:

    def test_margin_calculation(self):
        """Margin should be (price - cost) / price."""
        p = Product(
            product_id=0, name="test", category="produce",
            base_cost=3.00, base_retail_price=6.00,
            promo_eligible=True,
            seasonality=np.ones(365),
            brand_strength=0.5,
        )
        assert p.margin(6.00) == pytest.approx(0.50)
        assert p.margin(3.00) == pytest.approx(0.00)

    def test_priced_below_cost(self):
        p = Product(
            product_id=0, name="test", category="produce",
            base_cost=3.00, base_retail_price=6.00,
            promo_eligible=True,
            seasonality=np.ones(365),
            brand_strength=0.5,
        )
        assert p.is_priced_below_cost(2.99) is True
        assert p.is_priced_below_cost(3.00) is False
        assert p.is_priced_below_cost(3.01) is False


class TestProductCatalog:

    @pytest.fixture
    def catalog(self):
        return ProductCatalog(seed=42)

    def test_total_product_count(self, catalog):
        """Should generate exactly 50 products."""
        assert len(catalog) == 50

    def test_category_counts(self, catalog):
        """Each category should have the correct number of products."""
        expected = {
            "produce": 12,
            "dairy_eggs": 12,
            "snacks": 10,
            "beverages": 10,
            "frozen": 6,
        }
        for category, count in expected.items():
            assert len(catalog.get_category(category)) == count, \
                f"Expected {count} products in {category}"

    def test_prices_above_cost(self, catalog):
        """Every product's retail price should exceed its base cost."""
        for product in catalog.get_all_products():
            assert product.base_retail_price > product.base_cost, \
                f"Product {product.name} priced below cost"

    def test_seasonality_shape(self, catalog):
        """Seasonality curve should be 365 days."""
        for product in catalog.get_all_products():
            assert product.seasonality.shape == (365,), \
                f"Product {product.name} has wrong seasonality shape"

    def test_seasonality_bounds(self, catalog):
        """Seasonality multipliers should stay in clipped range."""
        for product in catalog.get_all_products():
            assert product.seasonality.min() >= 0.5
            assert product.seasonality.max() <= 2.5

    def test_promo_eligible_subset(self, catalog):
        """Promo eligible products should be a subset of all products."""
        all_ids = {p.product_id for p in catalog.get_all_products()}
        promo_ids = {p.product_id for p in catalog.get_promo_eligible()}
        assert promo_ids.issubset(all_ids)

    def test_summary_dataframe_shape(self, catalog):
        """Summary should return a DataFrame with 50 rows and correct columns."""
        df = catalog.summary()
        assert len(df) == 50
        expected_cols = {"product_id", "name", "category", "base_cost",
                         "base_retail_price", "margin_pct", "brand_strength",
                         "promo_eligible"}
        assert expected_cols.issubset(set(df.columns))

    def test_reproducibility(self):
        """Same seed should produce identical catalogs."""
        c1 = ProductCatalog(seed=99)
        c2 = ProductCatalog(seed=99)
        for pid in range(50):
            assert c1.get_product(pid).base_retail_price == \
                   c2.get_product(pid).base_retail_price

    def test_different_seeds_differ(self):
        """Different seeds should produce different catalogs."""
        c1 = ProductCatalog(seed=1)
        c2 = ProductCatalog(seed=2)
        prices1 = [c1.get_product(i).base_retail_price for i in range(50)]
        prices2 = [c2.get_product(i).base_retail_price for i in range(50)]
        assert prices1 != prices2

    def test_repr(self, catalog):
        assert "50 products" in repr(catalog)
        assert "5 categories" in repr(catalog)