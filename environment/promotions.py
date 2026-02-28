import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from utils.config_loader import load_env_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PromoWindow:
    """A scheduled promotional period."""
    name:       str
    start_day:  int      # day of year (0-364)
    end_day:    int
    multiplier: float    # demand multiplier during this window
    is_holiday: bool     # True for major holidays, False for regular promos


@dataclass
class DayInfo:
    """All promotion-related info for a given day — fed into RL observation."""
    day:                  int
    is_weekend:           bool
    is_holiday:           bool
    active_promos:        List[str]      # names of active promo windows
    demand_multiplier:    float          # combined multiplier for today
    days_to_next_promo:   int            # how many days until next promo starts
    next_promo_name:      str            # name of the upcoming promo
    next_promo_multiplier: float         # multiplier of the upcoming promo


class PromotionCalendar:
    """
    Manages the promotional calendar for the retail simulation.

    Tracks two types of demand boosts:
    1. Scheduled windows — holidays and seasonal events baked into
       the calendar (Black Friday, Christmas, July 4th, etc.)
       These are KNOWN to all agents in advance via the observation space.

    2. Weekend effect — automatic 1.347x multiplier on weekends,
       calibrated from Instacart EDA.

    Agents observe:
    - Whether today is a weekend / holiday
    - Which promos are currently active
    - Days until the next promo window
    - The multiplier of the upcoming promo

    This gives agents the information needed to:
    - Build up inventory before high-demand periods
    - Aggressively price during promo windows
    - Undercut competitors who haven't prepared
    """

    # Built-in holiday calendar (day ranges are 0-indexed)
    DEFAULT_PROMO_WINDOWS = [
    PromoWindow("valentines_day",  44,  44,  1.3, False),
        PromoWindow("easter",          90,  92,  1.4, False),
        PromoWindow("memorial_day",   144, 146,  1.3, False),
        PromoWindow("july_4th",       183, 185,  1.4, True),
        PromoWindow("labor_day",      244, 246,  1.3, False),
        PromoWindow("halloween",      299, 300,  1.3, False),
        PromoWindow("thanksgiving",   325, 327,  1.6, True),
        PromoWindow("black_friday",   328, 329,  2.0, True),
        PromoWindow("cyber_monday",   331, 331,  1.7, True),
        PromoWindow("christmas_week", 357, 364,  2.0, True),
        PromoWindow("new_years",        0,   1,  1.5, True),
    ]

    def __init__(self, config=None, seed: int = 42):
        self.config = config or load_env_config()
        self.rng    = np.random.default_rng(seed)

        self.weekend_multiplier = self.config.get(
            'promotions.weekend_demand_multiplier', 1.347
        )

        # build promo windows from config + defaults
        self.promo_windows = self._build_promo_windows()

        # precompute day lookup for fast access
        self._day_cache: Dict[int, DayInfo] = {}
        self._precompute_all_days()

        logger.info(
            f"PromotionCalendar initialized | "
            f"{len(self.promo_windows)} promo windows | "
            f"weekend_multiplier={self.weekend_multiplier}"
        )

    def get_day_info(self, day: int) -> DayInfo:
        """Get all promotion info for a given day. O(1) after precompute."""
        return self._day_cache[day % 365]

    def get_multiplier(self, day: int) -> float:
        """Quick access to just the demand multiplier for a day."""
        return self._day_cache[day % 365].demand_multiplier

    def is_weekend(self, day: int) -> bool:
        """Day 0 = Monday. Days 5,6 = Saturday, Sunday."""
        return (day % 7) in (5, 6)

    def days_until_next_promo(self, day: int) -> Tuple[int, Optional[PromoWindow]]:
        """
        Returns (days_until, next_promo_window).
        Looks ahead up to 365 days to find the next promo start.
        """
        for offset in range(1, 366):
            future_day = (day + offset) % 365
            for window in self.promo_windows:
                if future_day == window.start_day:
                    return offset, window
        return 365, None

    def get_observation_vector(self, day: int) -> np.ndarray:
        """
        Returns a compact numpy vector for the RL observation space.
        Shape: (5,)
            [is_weekend, is_holiday, demand_multiplier,
             days_to_next_promo_normalized, next_promo_multiplier]
        """
        info = self.get_day_info(day)
        days_norm = min(info.days_to_next_promo, 30) / 30.0  # normalize to [0,1]
        return np.array([
            float(info.is_weekend),
            float(info.is_holiday),
            info.demand_multiplier,
            days_norm,
            info.next_promo_multiplier,
        ], dtype=np.float32)

    # ── Private helpers ───────────────────────────────────────────

    def _build_promo_windows(self) -> List[PromoWindow]:
        """Build promo windows from defaults + config overrides."""
        windows = list(self.DEFAULT_PROMO_WINDOWS)

        config_promos = self.config.get('promotions.holiday_periods', [])
        for cp in config_promos:
            windows.append(PromoWindow(
                name=f"config_promo_{cp.get('start',0)}",
                start_day=cp.get('start', 0),
                end_day=cp.get('end', 0),
                multiplier=cp.get('multiplier', 1.5),
                is_holiday=True,
            ))
        return windows

    def _precompute_all_days(self):
        """Precompute DayInfo for all 365 days for O(1) lookup."""
        for day in range(365):
            weekend    = self.is_weekend(day)
            active     = self._get_active_promos(day)
            multiplier = self._compute_multiplier(day, weekend, active)
            days_to, next_promo = self.days_until_next_promo(day)

            self._day_cache[day] = DayInfo(
                day=day,
                is_weekend=weekend,
                is_holiday=any(p.is_holiday for p in active),
                active_promos=[p.name for p in active],
                demand_multiplier=multiplier,
                days_to_next_promo=days_to,
                next_promo_name=next_promo.name if next_promo else "",
                next_promo_multiplier=next_promo.multiplier if next_promo else 1.0,
            )

    def _get_active_promos(self, day: int) -> List[PromoWindow]:
        """Return all promo windows active on a given day."""
        active = []
        for window in self.promo_windows:
            if window.start_day <= window.end_day:
                if window.start_day <= day <= window.end_day:
                    active.append(window)
            else:
                # wraps around year end (e.g. new year)
                if day >= window.start_day or day <= window.end_day:
                    active.append(window)
        return active

    def _compute_multiplier(
        self, day: int, is_weekend: bool, active_promos: List[PromoWindow]
    ) -> float:
        """
        Compute combined demand multiplier.
        Takes the MAX of all active promo multipliers (not additive —
        Black Friday doesn't double-stack with Thanksgiving).
        Weekend multiplier stacks multiplicatively on top.
        """
        base = 1.0
        if active_promos:
            base = max(p.multiplier for p in active_promos)
        if is_weekend:
            base *= self.weekend_multiplier
        return round(base, 4)

    def __repr__(self) -> str:
        return (f"PromotionCalendar("
                f"{len(self.promo_windows)} windows, "
                f"weekend_mult={self.weekend_multiplier})")