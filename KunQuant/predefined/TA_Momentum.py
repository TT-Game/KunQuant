from KunQuant.Op import *
from KunQuant.ops import *
import numpy as np

class AllData:
    def __init__(self, open: OpBase, close: OpBase = None, high: OpBase = None, low: OpBase = None, volume: OpBase = None, amount: OpBase = None, vwap: OpBase = None) -> None:
        self.open = open
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.amount = amount
        if vwap is None:
            self.vwap = Div(self.amount, AddConst(self.volume, 0.0000001))
        else:
            self.vwap = vwap
        self.returns = returns(close)


def stddev(v: OpBase, window: int) -> OpBase:
    return WindowedStddev(v, window)

def returns(v: OpBase) -> OpBase:
    prev1 = BackRef(v, 1)
    return SubConst(Div(v, prev1), 1.0)

def ts_argmax(v: OpBase, window: int) -> OpBase:
    return TsArgMax(v, window)

def ts_argmin(v: OpBase, window: int) -> OpBase:
    return TsArgMin(v, window)

def ts_rank(v: OpBase, window: int) -> OpBase:
    return TsRank(v, window)

def ts_sum(v: OpBase, window: int) -> OpBase:
    return WindowedSum(v, window)

def ts_min(v: OpBase, window: int) -> OpBase:
    return WindowedMin(v, window)

def ts_max(v: OpBase, window: int) -> OpBase:
    return WindowedMax(v, window)

def correlation(v1: OpBase, v2: OpBase, window: int) -> OpBase:
    return WindowedCorrelation(v1, window, v2)

def delta(v1: OpBase, window: int = 1) -> OpBase:
    return Sub(v1, BackRef(v1, window))

def rank(v: OpBase)-> OpBase:
    return Rank(v)

def sign(v: OpBase)-> OpBase:
    return Sign(v)

def covariance(v: OpBase, v2: OpBase, window: int) -> OpBase:
    return WindowedCovariance(v, window, v2)

def ts_sma(v: OpBase, window: int) -> OpBase:
    return WindowedAvg(v, window)

def ts_ema(v: OpBase, window: int) -> OpBase:
    factor = factor.copy()
    sma_nth = factor[0:window].mean()
    factor[:window - 1] = np.nan
    factor.iloc[window - 1] = sma_nth
    ema = factor.ewm(span=window, adjust=False).mean()
    return ema

def ts_linreg(v: OpBase, window: int, tsf: bool, r: bool) -> OpBase:
    pass

def ts_rma(v: OpBase, window: int):
    pass
    # alpha = 1.0 / window
    # rma = factor.ewm(alpha=alpha, min_periods=window).mean()
    # return rma

def bool_to_10(v: OpBase) -> OpBase:
    return Select(v, ConstantOp(1), ConstantOp(0))

def safe_div(v: OpBase, divisor: OpBase) -> OpBase:
    divisor = Select(Equals(divisor, ConstantOp(0)), ConstantOp(0.0001), divisor)
    return v / divisor

def clip_lower(v: OpBase, lower: int) -> OpBase:
    return Select(v < lower, lower, v)

def clip_upper(v: OpBase, upper: int) -> OpBase:
    return Select(v > upper, upper, v)

def true_range_(high: OpBase, low: OpBase, close: OpBase):
    prev_close = ts_delay(close, 1)
    true_range = GreaterThan(GreaterThan(abs(high - low), abs(high - prev_close)), abs(prev_close - low))
    true_range.iloc[:1] = np.nan
    return true_range

def atr_(high: OpBase, low: OpBase, close: OpBase, window: int):
    tr = true_range_(high, low, close)
    atr = ts_rma(tr, window)
    return atr

ts_delay = BackRef

# ta_momentum factor
def ao_(d: AllData, fast: int, slow: int):
    median = (d.high + d.low) / 2
    value = ts_sma(median, fast) - ts_sma(median, slow)
    return value

def bias_(d: AllData, window: int):
    value = d.close / ts_sma(d.close, window) - 1
    return value

def brar_(d: AllData, window: int):
    ho_diff = d.high - d.open
    ol_diff = d.open - d.low
    hcy = d.high - ts_delay(d.close, 1)
    cyl = ts_delay(d.close, 1) - d.low
    hcy[hcy < 0] = 0
    cyl[cyl < 0] = 0
    ar = 100 * safe_div(ts_sum(ho_diff, window), ts_sum(ol_diff, window))
    br = 100 * safe_div(ts_sum(hcy, window), ts_sum(cyl, window))
    return ar, br

def cfo_(d: AllData, window: int):
    value = 100 * safe_div((d.close - ts_linreg(d.close, window, tsf=True)), d.close)
    return value

def cmo_(d: AllData, window: int):
    momp = delta(d.close, 1)
    momn = delta(d.close, 1)
    positive = clip_lower(momp, 0)
    negative = abs(clip_upper(momn, 0))
    pos_ = ts_sum(positive, window)
    neg_ = ts_sum(negative, window)
    value = 100 * safe_div(pos_ - neg_, pos_ + neg_)
    return value

def cti_(d: AllData, window: int):
    value = ts_linreg(d.close, window, r=True)
    return value

def er_(d: AllData, window: int):
    abs_diff = abs(delta(d.close, window))
    volatility = abs(delta(d.close, 1))
    value = safe_div(abs_diff, ts_sum(volatility, window))
    return value

def kdj_(d: AllData, window: int, signal: int):
    highest_high = ts_max(d.high, window)
    lowest_low = ts_min(d.low, window)
    fast_k = 100 * safe_div(d.close - lowest_low, highest_high - lowest_low)
    k = ts_rma(fast_k, signal)
    d = ts_rma(k, signal)
    j = 3 * k - 2 * d
    return k, d, j

def macd_(d: AllData, fast: int, slow: int, signal: int):
    macd = ts_ema(d.close, fast) - ts_ema(d.close, slow)
    signal_ma = ts_ema(macd.loc[macd.first_valid_index():, ], signal)
    histogram = macd - signal_ma
    return macd, signal_ma, histogram

def pgo_(d: AllData, window: int):
    pgo = d.close - ts_sma(d.close, window)
    pgo /= ts_ema(atr_(d.high, d.low, d.close, window), window)
    return pgo

def psl_(d: AllData, window: int):
    diff = sign(d.close - d.open)
    diff.fillna(0, inplace=True)
    diff[diff <= 0] = 0
    psl = 100 * ts_sum(diff, window)
    psl /= window
    return psl

def qqe_():
    pass

def rsi_(d: AllData, window: int):
    negative = delta(d.close, 1)
    positive = negative.copy()

    positive[positive < 0] = 0  # Make negatives 0 for the postive series
    negative[negative > 0] = 0  # Make postives 0 for the negative series

    positive_avg = ts_rma(positive, window)
    negative_avg = ts_rma(negative, window)

    rsi = 100 * safe_div(positive_avg, positive_avg + abs(negative_avg))
    return rsi


def rvgi_(d: AllData, window: int,
          swma_window: int):
    numerator = ts_sum(ts_swma(d.close - d.open, swma_window), window)
    denominator = ts_sum(ts_swma(d.high - d.low, swma_window), window)
    rvgi = safe_div(numerator, denominator)
    signal = ts_swma(rvgi, swma_window)
    return rvgi, signal


def slope_(d: AllData, window: int):
    slope = delta(d.close, window) / window
    return slope

def stochrsi_(d: AllData, window: int, rsi_window: int, k: int, d: int):
    rsi = rsi_(d.close, rsi_window)
    lowest_rsi = ts_min(rsi, window)
    highest_rsi = ts_max(rsi, window)
    stoch = 100 * (rsi - lowest_rsi)
    stoch = safe_div(stoch, highest_rsi - lowest_rsi)
    stochrsi_k = ts_sma(stoch, k)
    stochrsi_d = ts_sma(stochrsi_k, d)
    return stochrsi_k, stochrsi_d

def trix_(d: AllData, window: int, signal: int):
    ema1 = ts_ema(d.close, window)
    ema2 = ts_ema(ema1, window)
    ema3 = ts_ema(ema2, window)
    trix = 100 * ema3.pct_change(1)
    trix_signal = ts_sma(trix, signal)
    return trix, trix_signal

def uo_(d: AllData, fast: int, medium: int, slow: int):
    fast_w, medium_w, slow_w = 4.0, 2.0, 1.0
    min_l_or_pc = LessThan(d.low, ts_delay(d.close, 1))
    max_h_or_pc = GreaterThan(d.high, ts_delay(d.close, 1))
    bp = d.close - min_l_or_pc
    tr = max_h_or_pc - min_l_or_pc
    fast_avg = safe_div(ts_sum(bp, fast), ts_sum(tr, fast))
    medium_avg = safe_div(ts_sum(bp, medium), ts_sum(tr, medium))
    slow_avg = safe_div(ts_sum(bp, slow), ts_sum(tr, slow))
    total_weight = fast_w + medium_w + slow_w
    weights = (fast_w * fast_avg) + (medium_w * medium_avg) + (slow_w * slow_avg)
    uo = 100 * weights / total_weight
    return uo

def ao(d: AllData):
    fast_slow_list = [(5, 34), (10, 68), (20, 136), (30, 204), (60, 408), (120, 816),
                        (240, 1632), (480, 3264)]
    result_list = list()
    for fast, slow in fast_slow_list:
        value = ao_(d, fast, slow) / d.close
        result_list.append(("AO_%s_%s" % (fast, slow), value))

    return result_list

all_alpha = [ao]