import numpy as np
import pytest
from environment.promotions import PromotionCalendar, DayInfo, PromoWindow


@pytest.fixture
def cal():
    return PromotionCalendar()


class TestPromotionCalendarInit:

    def test_repr(self, cal):
        assert "windows" in repr(cal)
        assert "weekend_mult" in repr(cal)

    def test_precomputed_all_days(self, cal):
        assert len(cal._day_cache) == 365

    def test_weekend_multiplier_loaded(self, cal):
        assert cal.weekend_multiplier == pytest.approx(1.347)


class TestWeekendDetection:

    def test_weekday_not_weekend(self, cal):
        # day 0 = Monday
        assert cal.is_weekend(0) is False
        assert cal.is_weekend(1) is False
        assert cal.is_weekend(4) is False

    def test_weekend_days(self, cal):
        # day 5 = Saturday, day 6 = Sunday
        assert cal.is_weekend(5) is True
        assert cal.is_weekend(6) is True

    def test_weekend_multiplier_applied(self, cal):
        # find a weekend day with no promo
        info = cal.get_day_info(180)  # normal summer weekend
        assert info.is_weekend is True
        assert info.demand_multiplier == pytest.approx(1.347, abs=0.01)


class TestHolidayDetection:

    def test_new_years_is_holiday(self, cal):
        info = cal.get_day_info(0)
        assert info.is_holiday is True
        assert "new_years" in info.active_promos

    def test_black_friday_is_holiday(self, cal):
        info = cal.get_day_info(328)
        assert info.is_holiday is True
        assert "black_friday" in info.active_promos

    def test_christmas_week_is_holiday(self, cal):
        for day in range(357, 365):
            info = cal.get_day_info(day)
            assert info.is_holiday is True

    def test_normal_day_not_holiday(self, cal):
        info = cal.get_day_info(100)  # mid April, no promo
        assert info.is_holiday is False
        assert info.active_promos == []


class TestMultipliers:

    def test_black_friday_weekend_stacks(self, cal):
        # Black Friday (2.0) on a weekend (1.347) = 2.694
        info = cal.get_day_info(328)
        assert info.demand_multiplier == pytest.approx(2.0 * 1.347, abs=0.01)

    def test_normal_weekday_multiplier_is_one(self, cal):
        # find a normal weekday with no promo
        info = cal.get_day_info(100)
        assert info.demand_multiplier == pytest.approx(1.0, abs=0.01)

    def test_multiplier_always_positive(self, cal):
        for day in range(365):
            assert cal.get_multiplier(day) > 0

    def test_get_multiplier_matches_day_info(self, cal):
        for day in [0, 90, 183, 328, 360]:
            assert cal.get_multiplier(day) == cal.get_day_info(day).demand_multiplier


class TestNextPromo:

    def test_days_to_next_promo_positive(self, cal):
        for day in range(365):
            info = cal.get_day_info(day)
            assert info.days_to_next_promo > 0

    def test_next_promo_after_black_friday_is_cyber_monday(self, cal):
        info = cal.get_day_info(328)
        assert info.next_promo_name == "cyber_monday"
        assert info.days_to_next_promo == 3

    def test_next_promo_multiplier_positive(self, cal):
        for day in range(365):
            info = cal.get_day_info(day)
            assert info.next_promo_multiplier > 0


class TestObservationVector:

    def test_observation_vector_shape(self, cal):
        obs = cal.get_observation_vector(0)
        assert obs.shape == (5,)

    def test_observation_vector_dtype(self, cal):
        obs = cal.get_observation_vector(0)
        assert obs.dtype == np.float32

    def test_black_friday_observation(self, cal):
        obs = cal.get_observation_vector(328)
        assert obs[0] == 1.0   # is_weekend
        assert obs[1] == 1.0   # is_holiday
        assert obs[2] == pytest.approx(2.694, abs=0.01)  # multiplier

    def test_normal_weekday_observation(self, cal):
        obs = cal.get_observation_vector(100)
        assert obs[0] == 0.0   # not weekend
        assert obs[1] == 0.0   # not holiday
        assert obs[2] == pytest.approx(1.0, abs=0.01)