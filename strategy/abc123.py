import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata

# =============================== parameters ================================
EPS = 1e-8
MIN_OBS = 70
DEFAULT_DLR_LIMIT = 10_000.0
ALGO_DLR_LIMIT = 100_000.0

# Market-beta residuals.
BETA_CLIP = (0.25, 2.50)
BETA_SHORT_WINDOW = 250

# Lead-lag ridge forecast of the next residual.
LEADLAG_MIN_OBS = 30
RIDGE = 0.75
MARKET_LEADER_WEIGHT = 2.50

# Group-lasso leader/follower selection feeding the lead-lag refit.
LASSO_ALPHA = 0.25
LASSO_IDENTIFICATION_RIDGE = 0.01
LASSO_MAX_ITER = 300
LASSO_TOLERANCE = 1e-6
LASSO_ACTIVE_THRESHOLD = 1e-6

# Short-horizon residual reversal.
REV_WINS = (15, 20, 25)
REV_WEIGHT = 0.20
REV_NEUTRAL_PC = 2
REV_NEUTRAL_WINDOW = 250

# Cointegrated pair overlay.
PAIR_COUNT = 12
PAIR_MIN_FORMATION_OBS = 250
PAIR_EXTRA_START = 500
PAIR_RECENT_WINDOW = 500
PAIR_MID_SHORT_WINDOW = 360
PAIR_MID_LONG_WINDOW = 420
PAIR_REFRESH_DAYS = 60
PAIR_MIN_CORRELATION = 0.15
PAIR_MAX_AR1 = 0.98
PAIR_MIN_CROSSINGS = 8
PAIR_Z_WINDOWS = (100, 120, 150)
PAIR_GATE_Z = 0.40
PAIR_EXIT_Z = 0.35
PAIR_ACTIVE_CAP = 10
PAIR_CONVICTION_RATIO = 0.50
PAIR_BETA_LONG_WINDOW = 500
PAIR_BETA_SHORT_WINDOW = 250
PAIR_BETA_DRIFT_MAX = 0.40
PAIR_PROTECT_TOP_K = 2
PAIR_EARLY_OVERRIDE_CAP = 12
PAIR_FULL_OVERRIDE_CAP = 11

# Independent alphas blended on top of the core book.
LOWRANK_RIDGE = 1.6
LOWRANK_RANKS = (3, 4, 5, 6)
PAIRNET_TOP_K = 20
PAIRNET_ADF_MAX = -3.2
PAIRNET_Z_WINDOW = 250
PAIRNET_REFRESH = 50
LEVEL_HORIZONS = (250, 350, 500, 650)
LEVEL_TAIL_Z = 2.0

# Blender: rolling nonnegative logistic over the feature columns, whose order
# is fixed by BLEND_PRIOR_WEIGHTS and reproduced in _Strategy._blend_features.
PRIOR_CORE = 1.0
PRIOR_OVERLAY = 0.0
PRIOR_LOWRANK = 1.0
PRIOR_PAIRNET = 1.0
PRIOR_LEVEL = 0.25
PRIOR_NONLINEAR = 0.25
BLEND_PRIOR_WEIGHTS = np.asarray((
    PRIOR_CORE, PRIOR_OVERLAY, PRIOR_LOWRANK, PRIOR_PAIRNET, PRIOR_LEVEL,
    PRIOR_NONLINEAR, PRIOR_NONLINEAR,
))
BLEND_WINDOW = 125
BLEND_L2 = 0.001
BLEND_MAX_ITER = 50

# Instrument 0 leg: a capacity extension of the stock book, never its opposite.
ALGO_TILT_DLR = 100_000.0
ALGO_BREADTH_GATE = 0.04
ALGO_IDLE_MIN_NET_VOTES = 6
ALGO_IDLE_SEASONAL_LADDERS = ((22, 44, 88), (23, 46, 92))
ALGO_SEASONAL_MIN_OBS = max(max(l) for l in ALGO_IDLE_SEASONAL_LADDERS)


# ============================= generic helpers =============================
def _cs_z(x):
    """Cross-sectional z-score, clipped to keep single names from dominating."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    scale = np.std(x)
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros_like(x)
    return np.clip(x / scale, -4.0, 4.0)


def _position_limits(nins):
    limits = np.full(nins, DEFAULT_DLR_LIMIT)
    limits[0] = ALGO_DLR_LIMIT
    return limits


def _ols_beta_and_se(stocks, market):
    """Per-stock OLS beta on the market plus its conditional standard error."""
    market_centered = market - np.mean(market)
    stocks_centered = stocks - np.mean(stocks, axis=1, keepdims=True)
    denominator = float(market_centered @ market_centered) + EPS
    beta = stocks_centered @ market_centered / denominator
    errors = stocks_centered - beta[:, None] * market_centered[None, :]
    degrees = max(len(market) - 2, 1)
    variance = np.sum(errors * errors, axis=1) / degrees
    return beta, np.sqrt(variance / denominator)


def _signal_beta(stocks, market):
    """Drift-adaptive beta: move off the long-run beta only when the recent
    window differs by more than the sampling noise of the two estimates."""
    long_beta = np.clip(
        _ols_beta_and_se(stocks, market)[0], BETA_CLIP[0], BETA_CLIP[1]
    )
    if len(market) < 2 * BETA_SHORT_WINDOW:
        return long_beta
    recent_beta, recent_se = _ols_beta_and_se(
        stocks[:, -BETA_SHORT_WINDOW:], market[-BETA_SHORT_WINDOW:]
    )
    prior_beta, prior_se = _ols_beta_and_se(
        stocks[:, -2 * BETA_SHORT_WINDOW:-BETA_SHORT_WINDOW],
        market[-2 * BETA_SHORT_WINDOW:-BETA_SHORT_WINDOW],
    )
    difference = recent_beta - prior_beta
    noise_variance = recent_se * recent_se + prior_se * prior_se
    drift_weight = np.maximum(
        0.0, 1.0 - noise_variance / (difference * difference + EPS)
    )
    return np.clip(
        long_beta + drift_weight * (recent_beta - long_beta),
        BETA_CLIP[0],
        BETA_CLIP[1],
    )


def _residuals(stocks, market):
    """Beta-neutral residual returns, cross-sectionally demeaned."""
    resid = stocks - _signal_beta(stocks, market)[:, None] * market[None, :]
    return resid - np.mean(resid, axis=0, keepdims=True)


def _residual_vol(resid):
    return np.sqrt(np.mean(resid * resid, axis=1) + EPS * EPS)


_UPPER_INDEX_CACHE = {}


def _pair_upper_indices(nstocks):
    if nstocks not in _UPPER_INDEX_CACHE:
        _UPPER_INDEX_CACHE[nstocks] = np.triu_indices(nstocks, 1)
    return _UPPER_INDEX_CACHE[nstocks]


# ========================== group-lasso selection ==========================
def _group_soft_threshold(values, threshold, axis):
    norms = np.linalg.norm(values, axis=axis, keepdims=True)
    scale = np.maximum(0.0, 1.0 - threshold / np.maximum(norms, 1e-15))
    return values * scale


def _post_selection_ridge(xtx, xty, support):
    """Refit every follower on the market plus its selected residual leaders."""
    coefficients = np.zeros(xty.shape, dtype=float)
    for follower in range(xty.shape[1]):
        keep = np.concatenate((np.asarray((True,)), support[:, follower]))
        retained = int(np.sum(keep))
        coefficients[keep, follower] = np.linalg.solve(
            xtx[np.ix_(keep, keep)] + RIDGE * np.eye(retained),
            xty[keep, follower],
        )
    return coefficients


# ============================ core book signals ============================
def _residual_reversal(resid):
    """Multi-horizon residual reversal, neutralised against the leading PCs."""
    vol = _residual_vol(resid)
    reversal = np.zeros(resid.shape[0])
    for window in REV_WINS:
        width = min(window, resid.shape[1])
        reversal += -np.sum(resid[:, -width:], axis=1) / (
            vol * np.sqrt(width) + EPS
        )
    reversal /= len(REV_WINS)
    # Neutralising needs strictly more names than components, otherwise the
    # projection would remove the whole cross-section.
    if 0 < REV_NEUTRAL_PC < resid.shape[0]:
        try:
            width = min(REV_NEUTRAL_WINDOW, resid.shape[1])
            correlation = np.nan_to_num(
                np.corrcoef(resid[:, -width:]), nan=0.0
            )
            np.fill_diagonal(correlation, 1.0)
            loadings = np.linalg.eigh(correlation)[1][:, -REV_NEUTRAL_PC:]
            reversal = reversal - loadings @ (loadings.T @ reversal)
        except np.linalg.LinAlgError:
            pass
    return reversal


# =============================== alpha signals =============================
def _lowrank_lag1_signal(returns, market, resid):
    """Low-rank lag-1 propagation forecast from standardised raw returns."""
    if returns.shape[1] < 2:
        return np.zeros(returns.shape[0])
    predictors = np.vstack((returns, market[None, :]))
    predictors = (
        predictors - np.mean(predictors, axis=1, keepdims=True)
    ) / (np.std(predictors, axis=1, keepdims=True) + EPS)
    targets = (resid - np.mean(resid, axis=1, keepdims=True)) / (
        np.std(resid, axis=1, keepdims=True) + EPS
    )
    x = predictors[:, :-1].T
    y = targets[:, 1:].T
    gram = x.T @ x / len(x)
    cross = x.T @ y / len(x)
    try:
        coefficients = np.linalg.solve(
            gram + LOWRANK_RIDGE * np.eye(gram.shape[0]), cross
        )
        left, spectrum, right = np.linalg.svd(coefficients, full_matrices=False)
    except np.linalg.LinAlgError:
        return np.zeros(returns.shape[0])
    latest = predictors[:, -1]
    accumulated = np.zeros(returns.shape[0])
    for rank in LOWRANK_RANKS:
        accumulated += _cs_z(
            latest @ ((left[:, :rank] * spectrum[:rank]) @ right[:rank])
        )
    return _cs_z(accumulated)


def _level_reversion_signal(resid):
    """Long-horizon residual level reversion, averaged over several horizons
    so that no single lucky look-back is selected."""
    scale = np.sqrt(np.mean(resid * resid, axis=1)) + EPS
    accumulated = np.zeros(resid.shape[0])
    for horizon in LEVEL_HORIZONS:
        width = min(horizon, resid.shape[1])
        accumulated += _cs_z(
            -np.sum(resid[:, -width:], axis=1) / (scale * np.sqrt(width))
        )
    return _cs_z(accumulated)


def _adf_t_every_pair(log_prices):
    """Vectorised ADF(1) t-statistic for every pair spread (time-major)."""
    left, right = _pair_upper_indices(log_prices.shape[1])
    spread = log_prices[:, left] - log_prices[:, right]
    statistics = np.full(spread.shape[1], np.nan)
    change = np.diff(spread, axis=0)
    dependent = change[1:]
    level = spread[1:-1]
    lagged = change[:-1]
    rows = len(dependent)
    if rows <= 4:
        return statistics
    design = np.empty((spread.shape[1], 3, 3))
    design[:, 0, 0] = rows
    design[:, 0, 1] = design[:, 1, 0] = level.sum(0)
    design[:, 0, 2] = design[:, 2, 0] = lagged.sum(0)
    design[:, 1, 1] = (level * level).sum(0)
    design[:, 1, 2] = design[:, 2, 1] = (level * lagged).sum(0)
    design[:, 2, 2] = (lagged * lagged).sum(0)
    moment = np.stack(
        (
            dependent.sum(0),
            (level * dependent).sum(0),
            (lagged * dependent).sum(0),
        ),
        axis=1,
    )

    finite = (
        np.isfinite(design).all(axis=(1, 2))
        & np.isfinite(moment).all(axis=1)
    )
    screened = np.flatnonzero(finite)
    if len(screened) == 0:
        return statistics
    ranks = np.zeros(spread.shape[1], dtype=int)
    condition_numbers = np.full(spread.shape[1], np.inf)
    try:
        ranks[screened] = np.linalg.matrix_rank(design[screened])
        condition_numbers[screened] = np.linalg.cond(design[screened])
    except np.linalg.LinAlgError:
        for index in screened:
            try:
                ranks[index] = np.linalg.matrix_rank(design[index])
                condition_numbers[index] = np.linalg.cond(design[index])
            except np.linalg.LinAlgError:
                continue

    valid = np.flatnonzero(
        (ranks == design.shape[1])
        & np.isfinite(condition_numbers)
        & (condition_numbers <= 1e12)
    )
    if len(valid) == 0:
        return statistics

    try:
        coefficients = np.linalg.solve(
            design[valid], moment[valid, ..., None]
        )[..., 0]
        errors = dependent[:, valid] - (
            coefficients[:, 0][None, :]
            + level[:, valid] * coefficients[:, 1][None, :]
            + lagged[:, valid] * coefficients[:, 2][None, :]
        )
        variance = (errors * errors).sum(0) / (rows - 3)
        inverse = np.linalg.inv(design[valid])
        standard_error = np.sqrt(
            np.maximum(variance * inverse[:, 1, 1], 1e-30)
        )
        statistics[valid] = coefficients[:, 1] / standard_error
    except np.linalg.LinAlgError:
        for index in valid:
            try:
                coefficient = np.linalg.solve(design[index], moment[index])
                error = dependent[:, index] - (
                    coefficient[0]
                    + level[:, index] * coefficient[1]
                    + lagged[:, index] * coefficient[2]
                )
                variance = float(error @ error) / (rows - 3)
                inverse = np.linalg.inv(design[index])
                standard_error = np.sqrt(
                    max(variance * inverse[1, 1], 1e-30)
                )
                statistics[index] = coefficient[1] / standard_error
            except np.linalg.LinAlgError:
                continue
    return statistics


# ============================== pair overlay ===============================
def _ar1(spread):
    left = spread[:-1] - np.mean(spread[:-1])
    right = spread[1:] - np.mean(spread[1:])
    return float(left @ right / (left @ left + EPS))


def _adf_t_lag1(spread):
    delta = np.diff(spread)
    dependent = delta[1:]
    design = np.column_stack((
        np.ones(len(dependent)), spread[1:-1], delta[:-1]
    ))
    if len(dependent) <= design.shape[1]:
        return np.nan
    try:
        coefficient = np.linalg.lstsq(design, dependent, rcond=None)[0]
        error = dependent - design @ coefficient
        variance = float(error @ error) / (len(dependent) - design.shape[1])
        covariance = variance * np.linalg.inv(design.T @ design)
        standard_error = float(np.sqrt(covariance[1, 1]))
    except np.linalg.LinAlgError:
        return np.nan
    if not np.isfinite(standard_error) or standard_error <= 0:
        return np.nan
    return float(coefficient[1] / standard_error)


def _select_pairs(price_history, window=None):
    """Pick disjoint mean-reverting stock pairs from the visible prefix.

    Returns ``(adf_t, left, right)`` tuples, most stationary first.  Pairs are
    disjoint because a pair sharing a name with an accepted one is a derived
    combination that looks cointegrated in sample and stops reverting later.
    """
    log_prices = np.log(np.asarray(price_history, dtype=float).T)
    if window is not None:
        log_prices = log_prices[-min(int(window), len(log_prices)):]
    returns = np.diff(log_prices, axis=0)
    candidates = []
    for left_index in range(1, log_prices.shape[1]):
        for right_index in range(left_index + 1, log_prices.shape[1]):
            left = log_prices[:, left_index]
            right = log_prices[:, right_index]
            intercept = float(np.mean(left - right))
            spread = left - intercept - right
            spread_scale = float(np.std(spread))
            if not np.isfinite(spread_scale) or spread_scale <= EPS:
                continue
            phi = _ar1(spread)
            correlation = float(np.corrcoef(
                returns[:, left_index], returns[:, right_index]
            )[0, 1])
            crossings = int(np.count_nonzero(
                np.diff(np.sign(spread - np.mean(spread)))
            ))
            if (
                not np.isfinite(phi)
                or not 0.0 < phi < PAIR_MAX_AR1
                or not np.isfinite(correlation)
                or correlation <= PAIR_MIN_CORRELATION
                or crossings < PAIR_MIN_CROSSINGS
            ):
                continue
            adf_t = _adf_t_lag1(spread)
            if np.isfinite(adf_t):
                candidates.append((adf_t, left_index, right_index))
    candidates.sort()
    selected = []
    used = set()
    for candidate in candidates:
        if candidate[1] in used or candidate[2] in used:
            continue
        selected.append(candidate)
        used.update(candidate[1:])
        if len(selected) >= PAIR_COUNT:
            break
    return tuple(selected)


def _beta_structure_healthy(log_prices, left, right):
    """Reject a pair whose hedge ratio has drifted between the two windows."""
    long_width = min(PAIR_BETA_LONG_WINDOW, log_prices.shape[1])
    short_width = min(PAIR_BETA_SHORT_WINDOW, log_prices.shape[1])
    betas = []
    for width in (long_width, short_width):
        x = log_prices[right, -width:]
        y = log_prices[left, -width:]
        centered = x - np.mean(x)
        betas.append(float(
            centered @ (y - np.mean(y)) / (centered @ centered + EPS)
        ))
    return bool(
        np.isfinite(betas).all()
        and abs(betas[1] - betas[0]) <= PAIR_BETA_DRIFT_MAX
    )


class _PairBook:
    """One pair-selection window plus the live entry/exit state of its pairs."""

    __slots__ = ("window", "min_obs", "models", "states")

    def __init__(self, window, min_obs):
        self.window = window
        self.min_obs = min_obs
        self.models = ()
        self.states = {}

    def clear(self):
        self.models = ()
        self.states = {}

    def refit(self, price_history):
        self.models = _select_pairs(price_history, self.window)

    def target_sign(self, log_prices):
        """Per-instrument direction plus the z-score backing it.

        A pair is entered only when all z-windows agree on the sign and the
        median z clears the gate, and is dropped as soon as they disagree or
        the spread returns inside the exit band.
        """
        nins = log_prices.shape[0]
        if not self.models:
            return np.zeros(nins), np.zeros(nins)

        live_keys = {(model[1], model[2]) for model in self.models}
        for key in tuple(self.states):
            if key not in live_keys:
                del self.states[key]

        active = []
        for _, left, right in self.models:
            key = (left, right)
            if not _beta_structure_healthy(log_prices, left, right):
                self.states.pop(key, None)
                continue

            z_values = []
            for window in PAIR_Z_WINDOWS:
                width = min(window, log_prices.shape[1])
                spread = log_prices[left, -width:] - log_prices[right, -width:]
                rolling_scale = float(np.std(spread))
                if not np.isfinite(rolling_scale) or rolling_scale <= EPS:
                    z_values = []
                    break
                z_values.append(float(
                    (spread[-1] - np.mean(spread)) / (rolling_scale + EPS)
                ))
            if not z_values:
                self.states.pop(key, None)
                continue

            z_array = np.asarray(z_values)
            unanimous = bool(np.all(z_array > 0.0) or np.all(z_array < 0.0))
            z_score = float(np.median(z_array))
            desired_side = -int(np.sign(z_score))
            side = int(self.states.get(key, 0))

            if side == 0:
                if unanimous and abs(z_score) >= PAIR_GATE_Z:
                    side = desired_side
            elif not unanimous or abs(z_score) <= PAIR_EXIT_Z:
                side = 0
            elif desired_side != side:
                # Never hold the old direction once the spread has crossed
                # through equilibrium while inside the entry band.
                side = desired_side if abs(z_score) >= PAIR_GATE_Z else 0

            if side == 0:
                self.states.pop(key, None)
                continue
            self.states[key] = side
            active.append((abs(z_score), left, right, side))

        active.sort(key=lambda item: (-item[0], item[1], item[2]))
        target_dollars = np.zeros(nins)
        confidence = np.zeros(nins)
        for abs_z, left, right, side in active[:PAIR_ACTIVE_CAP]:
            target_dollars[left] += DEFAULT_DLR_LIMIT * side
            target_dollars[right] -= DEFAULT_DLR_LIMIT * side
            confidence[left] = max(confidence[left], abs_z)
            confidence[right] = max(confidence[right], abs_z)
        direction = np.sign(np.clip(
            target_dollars, -DEFAULT_DLR_LIMIT, DEFAULT_DLR_LIMIT
        ))
        return direction, confidence


# ================================ strategy =================================
class _Strategy:
    """Stateful engine driving :func:`getMyPosition`.

    Each decision is built in four stages:

    1. core book    - beta-neutral lead-lag ridge plus residual reversal;
    2. alphas       - a soft pair innovation, low-rank propagation,
                      pair-network pressure, level reversion and tails;
    3. blender      - a rolling nonnegative logistic over those columns decides
                      the sign held in every stock;
    4. instrument 0 - a capacity extension of the stock book, sized only when
                      the book is one-sided or would otherwise sit idle.

    Only the visible price prefix is ever used; no future price is read.
    """

    def __init__(self):
        self.books = (
            _PairBook(None, PAIR_MIN_FORMATION_OBS),
            _PairBook(PAIR_RECENT_WINDOW, PAIR_MIN_FORMATION_OBS),
            _PairBook(PAIR_MID_SHORT_WINDOW, PAIR_EXTRA_START),
            _PairBook(PAIR_MID_LONG_WINDOW, PAIR_EXTRA_START),
        )
        self._reset_core()
        # Blender state deliberately survives a core reset.
        self.blend_ready = False
        self.replaying = False
        self.replay_rows = []
        self.features = []
        self.targets = []
        self.weights = BLEND_PRIOR_WEIGHTS.copy()
        self.pending_features = None
        self.pending_nt = 0
        self.live_history = None

    def _reset_core(self):
        for book in self.books:
            book.clear()
        self.pair_fit_nt = 0
        self.lasso_row = None
        self.lasso_column = None
        self.lasso_support = None
        self.pairnet_fit_nt = None
        self.pairnet_pairs = None
        self.pairnet_weights = None
        self.last_nt = 0
        self.last_history = None
        self.last_output = None

    # ------------------------------------------------------------ entry --
    def step(self, prc_so_far):
        price_history = np.asarray(prc_so_far, dtype=float)
        nins, nt = price_history.shape

        if not self.replaying and nt >= MIN_OBS and (
            not self.blend_ready or not self._extends_live(price_history)
        ):
            self._rebuild_blend_history(price_history)

        if (
            self.last_output is not None
            and price_history.shape == self.last_history.shape
            and np.array_equal(price_history, self.last_history)
        ):
            return self.last_output.copy()

        if nt <= self.last_nt or not self._extends_last(price_history):
            self._reset_core()
        self.last_nt = nt

        prices = price_history[:, -1]
        if nt < MIN_OBS or nins < 2 or np.any(prices <= 0.0):
            return self._remember(price_history, np.zeros(nins, dtype=int))

        # The current last price labels yesterday's counterfactual core books
        # before today's pair decision is made.

        limits = _position_limits(nins)
        share_limits = (limits / prices).astype(int)
        self._refresh_pair_books(price_history, nt)

        returns = np.diff(np.log(price_history), axis=1)
        market = returns[0]
        stocks = returns[1:]
        resid = _residuals(stocks, market)

        base_core_pos, raw_overlay_pos = self._core_books(
            price_history, prices, limits, share_limits, market, resid
        )
        core_side = np.sign(base_core_pos[1:])
        overlay_innovation = 0.5 * (
            np.sign(raw_overlay_pos[1:]) - core_side
        )
        features, prior_side = self._blend_features(
            price_history, nt, core_side, overlay_innovation,
            market, stocks, resid
        )

        if self.replaying:
            # The warm-up replay exists only to relabel past feature rows.
            self.replay_rows.append((nt, features))
            return self._remember(price_history, base_core_pos)

        algo_pos = _algo_tilt_position(
            core_side, prior_side, prices[0], share_limits[0]
        )
        self._label_pending(price_history)
        side = np.sign(features @ self._fit_weights())
        side = np.where(side == 0.0, core_side, side)

        positions = np.zeros(nins, dtype=int)
        positions[1:] = np.clip(
            (limits[1:] * side / prices[1:]).astype(int),
            -share_limits[1:],
            share_limits[1:],
        )
        positions[0] = algo_pos
        if positions[0] == 0:
            positions[0] = _idle_algo_position(side, market, share_limits[0])
        if positions[0] == 0 and price_history.shape[1] >= 2:
            latest_stock_returns = np.log(
                price_history[1:, -1] / price_history[1:, -2]
            )
            completion_side = int(np.sign(np.median(latest_stock_returns)))
            if completion_side != 0:
                positions[0] = completion_side * share_limits[0]

        self.pending_features = features.copy()
        self.pending_nt = nt
        self.live_history = price_history.copy()
        return self._remember(price_history, positions)

    def _remember(self, price_history, positions):
        self.last_history = price_history.copy()
        self.last_output = positions
        return positions.copy()

    def _extends_last(self, price_history):
        last = self.last_history
        if last is None:
            return True
        return bool(
            price_history.shape[0] == last.shape[0]
            and price_history.shape[1] > last.shape[1]
            and np.array_equal(price_history[:, :last.shape[1]], last)
        )

    def _extends_live(self, price_history):
        live = self.live_history
        if live is None:
            return False
        if price_history.shape == live.shape:
            return bool(np.array_equal(price_history, live))
        return bool(
            price_history.shape[0] == live.shape[0]
            and price_history.shape[1] == live.shape[1] + 1
            and np.array_equal(price_history[:, :live.shape[1]], live)
        )

    # -------------------------------------------------------- core book --
    def _core_books(self, price_history, prices, limits, share_limits,
                    market, resid):
        """Return stable base and raw pair-overridden core policies."""
        signal, gate = self._lead_lag_signals(market, resid)
        reversal = REV_WEIGHT * _cs_z(_residual_reversal(resid))
        signal = _cs_z(signal + reversal)
        gate = _cs_z(gate + reversal)
        book = np.zeros(len(prices), dtype=int)
        book[1:] = np.clip(
            (limits[1:] * np.sign(signal) / prices[1:]).astype(int),
            -share_limits[1:],
            share_limits[1:],
        )
        raw_overlay = self._apply_pair_overlay(
            price_history, prices, book, share_limits, gate
        )
        return book, raw_overlay

    def _lead_lag_signals(self, market, resid):
        """Ridge forecast of tomorrow's residual from today's leaders.

        Returns the sparse post-selection signal that sizes the book and the
        dense signal used as the conviction gate for the pair overlay.
        """
        empty = np.zeros(resid.shape[0])
        rows = resid.shape[1] - 1
        if rows < LEADLAG_MIN_OBS:
            return empty, empty.copy()
        leaders = np.vstack((market[None, :], resid))
        x_raw = leaders[:, -(rows + 1):-1].T
        y_raw = resid[:, -rows:].T
        x_mean = np.mean(x_raw, axis=0)
        x_std = np.std(x_raw, axis=0) + EPS
        x = (x_raw - x_mean) / x_std
        y = (y_raw - np.mean(y_raw, axis=0)) / (np.std(y_raw, axis=0) + EPS)
        xtx = (x.T @ x) / rows
        xty = (x.T @ y) / rows
        try:
            dense = np.linalg.solve(xtx + RIDGE * np.eye(xtx.shape[0]), xty)
            sparse = _post_selection_ridge(
                xtx, xty, self._group_lasso_support(resid)
            )
        except np.linalg.LinAlgError:
            return empty, empty.copy()
        sparse[0] *= MARKET_LEADER_WEIGHT
        dense[0] *= MARKET_LEADER_WEIGHT
        latest = (leaders[:, -1] - x_mean) / x_std
        return _cs_z(latest @ sparse), _cs_z(latest @ dense)

    def _group_lasso_support(self, residuals):
        """Select latent leader/follower groups from the visible prefix.

        Row and column group penalties are fitted with accelerated proximal
        descent; the union of active rows and columns becomes the support that
        the post-selection ridge is refitted on.
        """
        nstocks = residuals.shape[0]
        full_support = np.ones((nstocks, nstocks), dtype=bool)
        fallback = (
            self.lasso_support.copy()
            if self.lasso_support is not None
            and self.lasso_support.shape == full_support.shape
            else full_support
        )
        if residuals.shape[1] < 2 or not np.isfinite(residuals).all():
            return fallback

        x_raw = residuals[:, :-1].T
        y_raw = residuals[:, 1:].T
        x = (x_raw - np.mean(x_raw, axis=0)) / (np.std(x_raw, axis=0) + EPS)
        y = (y_raw - np.mean(y_raw, axis=0)) / (np.std(y_raw, axis=0) + EPS)
        xtx = (x.T @ x) / len(x)
        xty = (x.T @ y) / len(x)

        if (
            self.lasso_row is None
            or self.lasso_row.shape != xty.shape
            or self.lasso_column.shape != xty.shape
        ):
            row = np.zeros_like(xty)
            column = np.zeros_like(xty)
        else:
            row = np.asarray(self.lasso_row, dtype=float).copy()
            column = np.asarray(self.lasso_column, dtype=float).copy()
        np.fill_diagonal(row, 0.0)
        np.fill_diagonal(column, 0.0)
        accelerated_row = row.copy()
        accelerated_column = column.copy()
        momentum = 1.0

        try:
            base_lipschitz = float(np.linalg.eigvalsh(xtx)[-1] + RIDGE)
        except np.linalg.LinAlgError:
            return fallback
        step = 1.0 / max(
            2.0 * base_lipschitz + LASSO_IDENTIFICATION_RIDGE, 1e-12
        )
        for _ in range(LASSO_MAX_ITER):
            total = accelerated_row + accelerated_column
            common_gradient = xtx @ total - xty + RIDGE * total
            row_proposal = accelerated_row - step * (
                common_gradient + LASSO_IDENTIFICATION_RIDGE * accelerated_row
            )
            column_proposal = accelerated_column - step * (
                common_gradient
                + LASSO_IDENTIFICATION_RIDGE * accelerated_column
            )
            np.fill_diagonal(row_proposal, 0.0)
            np.fill_diagonal(column_proposal, 0.0)
            updated_row = _group_soft_threshold(
                row_proposal, step * LASSO_ALPHA, axis=1
            )
            updated_column = _group_soft_threshold(
                column_proposal, step * LASSO_ALPHA, axis=0
            )
            np.fill_diagonal(updated_row, 0.0)
            np.fill_diagonal(updated_column, 0.0)
            change = np.sqrt(
                np.linalg.norm(updated_row - row) ** 2
                + np.linalg.norm(updated_column - column) ** 2
            ) / max(
                1.0,
                np.sqrt(
                    np.linalg.norm(row) ** 2 + np.linalg.norm(column) ** 2
                ),
            )
            new_momentum = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * momentum * momentum))
            acceleration = (momentum - 1.0) / new_momentum
            accelerated_row = updated_row + acceleration * (updated_row - row)
            accelerated_column = (
                updated_column + acceleration * (updated_column - column)
            )
            row = updated_row
            column = updated_column
            momentum = new_momentum
            if change < LASSO_TOLERANCE:
                break

        if not np.isfinite(row).all() or not np.isfinite(column).all():
            return fallback
        self.lasso_row = row
        self.lasso_column = column
        active_rows = np.linalg.norm(row, axis=1) > LASSO_ACTIVE_THRESHOLD
        active_columns = np.linalg.norm(column, axis=0) > LASSO_ACTIVE_THRESHOLD
        self.lasso_support = active_rows[:, None] | active_columns[None, :]
        return self.lasso_support.copy()

    # ------------------------------------------------------ pair overlay --
    def _refresh_pair_books(self, price_history, nt):
        if nt < PAIR_MIN_FORMATION_OBS:
            return
        due = (
            not self.books[0].models
            or nt - self.pair_fit_nt >= PAIR_REFRESH_DAYS
        )
        for book in self.books:
            if nt >= book.min_obs and (due or not book.models):
                book.refit(price_history)
        if due:
            self.pair_fit_nt = nt

    def _apply_pair_overlay(self, price_history, prices, book, share_limits,
                            gate_signal):
        """Let the pair books flip a bounded number of core-book names.

        A name is flipped only when the pair z-score that supports it beats the
        core conviction on that name; the strongest core names are protected
        outright and the number of flips is capped.
        """
        if not any(pair_book.models for pair_book in self.books):
            return book

        log_prices = np.log(price_history)
        directions = []
        confidences = []
        for pair_book in self.books:
            direction, confidence = pair_book.target_sign(log_prices)
            directions.append(direction)
            confidences.append(confidence)
        combined_sign = np.sign(np.sum(np.asarray(directions), axis=0))

        pair_confidence = np.zeros(log_prices.shape[0])
        for index in np.flatnonzero(combined_sign):
            supporting = [
                confidence[index]
                for direction, confidence in zip(directions, confidences)
                if direction[index] == combined_sign[index]
                and confidence[index] > 0.0
            ]
            if supporting:
                pair_confidence[index] = min(supporting)

        pair_pos = np.clip(
            (combined_sign * DEFAULT_DLR_LIMIT / prices).astype(int),
            -share_limits,
            share_limits,
        )
        protected = set(
            (1 + np.argsort(-np.abs(gate_signal))[:PAIR_PROTECT_TOP_K]).tolist()
        )
        use_conviction_gate = price_history.shape[1] >= PAIR_EXTRA_START

        candidates = []
        for index in np.flatnonzero(combined_sign):
            if index in protected:
                continue
            required = (
                PAIR_CONVICTION_RATIO * abs(gate_signal[index - 1])
                if use_conviction_gate else 0.0
            )
            if pair_confidence[index] < required:
                continue
            if np.sign(pair_pos[index]) != np.sign(book[index]):
                candidates.append(
                    (float(pair_confidence[index] - required), int(index))
                )
        candidates.sort(key=lambda item: (-item[0], item[1]))

        active_books = sum(bool(b.models) for b in self.books)
        override_cap = (
            PAIR_EARLY_OVERRIDE_CAP if active_books <= 2
            else PAIR_FULL_OVERRIDE_CAP
        )
        result = book.copy()
        for _, index in candidates[:override_cap]:
            result[index] = pair_pos[index]
        return result

    # ---------------------------------------------------- alpha features --
    def _pairnet_signal(self, log_prices, nt):
        """Aggregate reversion pressure from the strongest cointegrating pairs.

        The pair set is refreshed on a slow cycle and kept disjoint; each pair
        pushes both of its legs by its own z-score weighted by its ADF t-stat.
        """
        nstocks = log_prices.shape[1]
        left, right = _pair_upper_indices(nstocks)
        if (
            self.pairnet_pairs is None
            or nt - self.pairnet_fit_nt >= PAIRNET_REFRESH
        ):
            statistics = _adf_t_every_pair(log_prices)
            eligible = np.flatnonzero(
                np.isfinite(statistics) & (statistics < PAIRNET_ADF_MAX)
            )
            selected = []
            claimed = set()
            for candidate in eligible[np.argsort(statistics[eligible])]:
                first = int(left[candidate])
                second = int(right[candidate])
                if first in claimed or second in claimed:
                    continue
                selected.append(candidate)
                claimed.update((first, second))
                if len(selected) >= PAIRNET_TOP_K:
                    break
            self.pairnet_pairs = np.asarray(selected, dtype=int)
            self.pairnet_weights = np.abs(statistics[self.pairnet_pairs])
            self.pairnet_fit_nt = nt
        if len(self.pairnet_pairs) == 0:
            return np.zeros(nstocks)
        window = log_prices[-min(PAIRNET_Z_WINDOW, len(log_prices)):]
        pairs = self.pairnet_pairs
        spread = window[:, left[pairs]] - window[:, right[pairs]]
        z_score = (spread[-1] - np.mean(spread, axis=0)) / (
            np.std(spread, axis=0) + EPS
        )
        pressure = np.zeros(nstocks)
        np.add.at(pressure, left[pairs], -z_score * self.pairnet_weights)
        np.add.at(pressure, right[pairs], z_score * self.pairnet_weights)
        return _cs_z(pressure)

    def _blend_features(self, price_history, nt, core_side,
                        overlay_innovation, market, stocks, resid):
        """Build the blender's design row and the prior-weighted book.

        The instrument-0 breadth veto intentionally uses only core, low-rank,
        pair-network and level signals.  The nonlinear level-tail and ranked
        pair-network columns are fitted by the blender but excluded from this
        separate, more conservative veto prior.
        """
        lowrank = _lowrank_lag1_signal(stocks, market, resid)
        pairnet = self._pairnet_signal(np.log(price_history[1:]).T, nt)
        level = _level_reversion_signal(resid)
        level_tail = np.where(np.abs(level) >= LEVEL_TAIL_Z, level, 0.0)
        # Smooth monotone tail basis: absolute cross-sectional ranks are scale
        # invariant, which avoids fitting a hard pair-network threshold.
        pairnet_rank = pairnet * (
            rankdata(np.abs(pairnet), method="average") / len(pairnet)
        )
        # Column order must match BLEND_PRIOR_WEIGHTS.
        features = np.column_stack((
            core_side, overlay_innovation, lowrank, pairnet, level,
            level_tail, pairnet_rank
        ))

        prior_score = (
            PRIOR_LOWRANK * lowrank
            + PRIOR_PAIRNET * pairnet
            + PRIOR_LEVEL * level
        ) + PRIOR_CORE * core_side
        prior_side = np.sign(prior_score)
        prior_side = np.where(prior_side == 0.0, core_side, prior_side)
        return features, prior_side

    # ---------------------------------------------------------- blender --
    def _rebuild_blend_history(self, price_history):
        """Causally replay the prefix to relabel the last training rows.

        Every replayed decision uses only prices strictly before it, and each
        feature row is stored with the day it was decided on so that labels
        cannot drift out of alignment if a day is skipped.
        """
        nt = price_history.shape[1]
        self.replay_rows = []
        self._reset_core()
        self.replaying = True
        try:
            for decision_nt in range(MIN_OBS, nt):
                self.step(price_history[:, :decision_nt])
        finally:
            self.replaying = False
        # The full prefix contains the outcome of the final replayed decision.
        rows = self.replay_rows[-BLEND_WINDOW:]
        self.features = [features.copy() for _, features in rows]
        self.targets = [
            np.sign(
                price_history[1:, decision_nt]
                / price_history[1:, decision_nt - 1]
                - 1.0
            )
            for decision_nt, _ in rows
        ]
        self.replay_rows = []
        self.weights = BLEND_PRIOR_WEIGHTS.copy()
        self.pending_features = None
        self.pending_nt = 0
        # Keep the model vintages reconstructed by the causal replay.  Resetting
        # here anchors 60/50-day refresh phases to the evaluator's first call.
        self.blend_ready = True

    def _label_pending(self, price_history):
        """Label the previous decision with the return that has just printed."""
        nt = price_history.shape[1]
        if self.pending_features is None or nt != self.pending_nt + 1:
            return
        realised = price_history[1:, -1] / price_history[1:, -2] - 1.0
        self.features.append(self.pending_features.copy())
        self.targets.append(np.sign(realised))
        self.features = self.features[-BLEND_WINDOW:]
        self.targets = self.targets[-BLEND_WINDOW:]

    def _fit_weights(self):
        """Refit the nonnegative rolling logistic, warm started at the prior.

        Nonnegativity keeps every column pointing the way it was designed to,
        so the fit can only reweight the alphas, never invert them.
        """
        if len(self.features) < BLEND_WINDOW:
            self.weights = BLEND_PRIOR_WEIGHTS.copy()
            return self.weights.copy()
        x = np.asarray(self.features[-BLEND_WINDOW:], dtype=float).reshape(
            -1, len(BLEND_PRIOR_WEIGHTS)
        )
        y = np.asarray(self.targets[-BLEND_WINDOW:], dtype=float).reshape(-1)
        scale = np.sqrt(np.mean(x * x, axis=0)) + 1e-12
        scaled = x / scale

        def objective(beta):
            margin = y * (scaled @ beta)
            loss = float(
                np.mean(np.logaddexp(0.0, -margin))
                + 0.5 * BLEND_L2 * (beta @ beta)
            )
            gradient = (
                -(scaled.T @ (y * expit(-margin))) / len(y) + BLEND_L2 * beta
            )
            return loss, gradient

        result = minimize(
            objective,
            np.maximum(self.weights * scale, 0.0),
            method="L-BFGS-B",
            jac=True,
            bounds=((0.0, None),) * len(BLEND_PRIOR_WEIGHTS),
            options={"maxiter": BLEND_MAX_ITER, "ftol": 1e-11, "gtol": 1e-8},
        )
        weights = np.maximum(result.x, 0.0) / scale
        if (
            result.success
            and np.all(np.isfinite(weights))
            and np.sum(weights) >= 1e-14
        ):
            self.weights = weights
        return self.weights.copy()


# ============================== instrument 0 ===============================
def _breadth_direction(side):
    breadth = float(np.mean(side))
    return float(np.sign(breadth)) if abs(breadth) >= ALGO_BREADTH_GATE else 0.0


def _algo_tilt_position(core_side, prior_side, price, share_limit):
    """Size instrument 0 along the book the stocks are already expressing.

    The core book is the lower-variance breadth estimate, so it does the
    sizing; the prior book only holds a veto.  A balanced prior book opposes
    nothing and therefore does not veto - only a prior book pointing the other
    way does.
    """
    core_direction = _breadth_direction(core_side)
    prior_direction = _breadth_direction(prior_side)
    if prior_direction == 0.0 or prior_direction == core_direction:
        tilt = core_direction
    else:
        tilt = 0.0
    dollars = float(np.clip(
        ALGO_TILT_DLR * tilt, -ALGO_DLR_LIMIT, ALGO_DLR_LIMIT
    ))
    return int(np.clip(int(dollars / price), -share_limit, share_limit))


def _idle_algo_position(side, market, share_limit):
    """Use otherwise-idle instrument 0 capacity, in that order of priority.

    First a clear directional majority in the stock book extends the same view
    through instrument 0.  Only if that is silent do two adjacent dyadic
    seasonal ladders get to speak, and only when both agree: each ladder votes
    across three time scales, and ``market[-lag]`` is observable now and refers
    to the return exactly ``lag`` days before the one being forecast.
    """
    net_votes = int(np.sum(side))
    if abs(net_votes) >= ALGO_IDLE_MIN_NET_VOTES:
        return int(np.sign(net_votes) * share_limit)
    if len(market) < ALGO_SEASONAL_MIN_OBS:
        return 0
    ladder_sides = [
        int(np.sign(sum(int(np.sign(market[-lag])) for lag in ladder)))
        for ladder in ALGO_IDLE_SEASONAL_LADDERS
    ]
    if ladder_sides[0] != 0 and all(s == ladder_sides[0] for s in ladder_sides):
        return int(ladder_sides[0] * share_limit)
    return 0


_STRATEGY = _Strategy()


def getMyPosition(prcSoFar):
    return _STRATEGY.step(prcSoFar)
