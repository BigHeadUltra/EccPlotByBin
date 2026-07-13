import os
import struct
import sys
import configparser
from datetime import datetime
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.signal import find_peaks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ---------- 中文字体 (SimHei) + 数学符号用 mathtext ----------
_font_path = "C:/Windows/Fonts/simhei.ttf"
try:
    font_manager.fontManager.addfont(_font_path)
    _prop = font_manager.FontProperties(fname=_font_path)
    matplotlib.rcParams["font.family"] = _prop.get_name()
except Exception:
    pass
matplotlib.rcParams["axes.unicode_minus"] = False

def _pause():
    """交互式暂停；非交互环境(管道/重定向)下静默退出，避免 EOFError。"""
    try:
        input("Press Enter to exit...")
    except (EOFError, KeyboardInterrupt):
        pass

CONFIG_FILE = 'config.ini'
DEFAULT_CONFIG = {
    'MAD_MULTIPLIER': '3.0',
    'STD_MULTIPLIER': '3.0',
    'MAD_SCALE': '1.4826',
    'MAX_VALUE': '1000'
}

def load_config():
    config = configparser.ConfigParser()
    config_updated = False
    
    if os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)
        
        if 'SETTINGS' not in config:
            config['SETTINGS'] = {}
        
        settings = config['SETTINGS']
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in settings:
                settings[key] = default_value
                config_updated = True
        
        if config_updated:
            with open(CONFIG_FILE, 'w') as f:
                config.write(f)
            print(f"Config file updated with missing items")
    else:
        config['SETTINGS'] = DEFAULT_CONFIG
        with open(CONFIG_FILE, 'w') as f:
            config.write(f)
        print(f"Config file not found, created default: {CONFIG_FILE}")
    
    settings = config['SETTINGS']
    return {
        'mad_multiplier': float(settings.get('MAD_MULTIPLIER', '3.0')),
        'std_multiplier': float(settings.get('STD_MULTIPLIER', '3.0')),
        'mad_scale': float(settings.get('MAD_SCALE', '1.4826')),
        'max_value': int(settings.get('MAX_VALUE', '1000'))
    }

def find_bin_files():
    bin_files = []
    for file in os.listdir('.'):
        if file.endswith('.bin'):
            bin_files.append(file)
    return bin_files

def determine_group_size(filename):
    if '1k' in filename.lower():
        return 1
    return 2

def read_bin_file(file_path, group_size, max_value=1000):
    data = []
    over_data = []
    invalid_value = 0xFF if group_size == 1 else 0xFFFF
    
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(group_size)
                if not chunk:
                    break
                if len(chunk) < group_size:
                    chunk = chunk.ljust(group_size, b'\x00')
                
                if group_size == 1:
                    value = chunk[0]
                else:
                    value = struct.unpack('<H', chunk)[0]
                
                if value != invalid_value:
                    if value <= max_value:
                        data.append(value)
                    else:
                        over_data.append(value)
        
        return data, over_data
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None

def welford_online_mean_std(data):
    n = 0
    mean = 0.0
    M2 = 0.0
    
    for x in data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    
    if n < 2:
        return mean, 0.0
    
    variance = M2 / (n - 1)
    return mean, variance ** 0.5

def count_statistics(data):
    if not data:
        return None
    
    max_val = max(data)
    counts = [0] * (max_val + 1)
    
    n = 0
    mean = 0.0
    M2 = 0.0
    
    for value in data:
        counts[value] += 1
        
        n += 1
        delta = value - mean
        mean += delta / n
        delta2 = value - mean
        M2 += delta * delta2
    
    std = (M2 / (n - 1)) ** 0.5 if n > 1 else 0.0
    
    return {
        'counts': counts,
        'total': n,
        'mean': mean,
        'std': std
    }

def calculate_thresholds(stats, config):
    mean_plus_std = stats['mean'] + config['std_multiplier'] * stats['std']
    
    return {
        'mean': stats['mean'],
        'mean_plus_std': mean_plus_std
    }

def calculate_percentages(counts, thresholds):
    total = sum(counts)
    
    def count_below(value):
        idx = int(value)
        if idx >= len(counts):
            return total
        return sum(counts[:idx+1])
    
    return {
        'mean_pct': (count_below(thresholds['mean']) / total) * 100,
        'mean_plus_std_pct': (count_below(thresholds['mean_plus_std']) / total) * 100
    }

def fit_distributions(counts):
    max_bit = min(len(counts) - 1, 1000)
    counts = counts[:max_bit + 1]
    
    start_with_zero = counts[0] > 0
    
    first_non_zero = None
    for value, count in enumerate(counts):
        if count > 0:
            first_non_zero = value
            break
    
    values = []
    weights = []
    for value, count in enumerate(counts):
        if value > 0 and count > 0:
            values.append(value)
            weights.append(count)
    
    if len(values) < 10:
        return None, None
    
    values = np.array(values)
    weights = np.array(weights)
    total = np.sum(weights)
    
    if start_with_zero:
        loc = 0.0
        x_shifted = values
    else:
        loc = first_non_zero - 1.0
        x_shifted = values - loc
    
    results = {}
    
    # 1. Normal distribution
    try:
        mean_w = np.sum(values * weights) / total
        var_w = np.sum(weights * (values - mean_w)**2) / total
        std_w = np.sqrt(var_w)
        results['norm'] = {
            'params': (mean_w, std_w),
            'score': -np.sum(weights * stats.norm.logpdf(values, mean_w, std_w)),
            'loc': loc
        }
    except:
        pass
    
    # 2. Lognormal distribution
    try:
        log_values = np.log(values[values > 0])
        log_weights = weights[values > 0]
        log_total = np.sum(log_weights)
        log_mean = np.sum(log_values * log_weights) / log_total
        log_var = np.sum(log_weights * (log_values - log_mean)**2) / log_total
        log_std = np.sqrt(log_var)
        results['lognorm'] = {
            'params': (log_std, 0, np.exp(log_mean)),
            'score': -np.sum(weights * stats.lognorm.logpdf(values, log_std, 0, np.exp(log_mean))),
            'loc': loc
        }
    except:
        pass
    
    # 3. Gamma distribution
    try:
        def neg_log_likelihood_gamma(params, x, w):
            a, scale = params
            if a <= 0 or scale <= 0:
                return 1e10
            return -np.sum(w * stats.gamma.logpdf(x, a, loc=0, scale=scale))
        
        mean_w = np.sum(values * weights) / total
        var_w = np.sum(weights * (values - mean_w)**2) / total
        a0 = mean_w**2 / var_w if var_w > 0 else 1.0
        scale0 = var_w / mean_w if mean_w > 0 else 1.0
        
        res = minimize(neg_log_likelihood_gamma, [a0, scale0], args=(x_shifted, weights), bounds=[(0.1, 100), (0.1, 1e6)])
        a_hat, scale_hat = res.x
        results['gamma'] = {
            'params': (a_hat, 0, scale_hat),
            'score': res.fun,
            'loc': loc
        }
    except:
        pass
    
    return results, loc

# ============================================================
# EM mixture 3-dist (Gamma + Gamma + Lognormal) — Fig4: mixture_fit.png
# ============================================================
def fit_mixture_em_3dist(counts, max_iter=100, tol=1e-6):
    """
    EM algorithm for 3-component mixture: Gamma + Gamma + Lognormal
    """
    max_bit = min(len(counts) - 1, 1000)
    counts = counts[:max_bit + 1]
    
    start_with_zero = counts[0] > 0
    
    first_non_zero = None
    for value, count in enumerate(counts):
        if count > 0:
            first_non_zero = value
            break
    
    values = []
    weights_data = []
    for value, count in enumerate(counts):
        if value > 0 and count > 0:
            values.append(value)
            weights_data.append(count)
    
    if len(values) < 10:
        return None
    
    values = np.array(values)
    weights_data = np.array(weights_data)
    
    if start_with_zero:
        loc = 0.0
        x_shifted = values
    else:
        loc = first_non_zero - 1.0
        x_shifted = values - loc
    
    log_values = np.log(x_shifted)
    log_mean = np.sum(log_values * weights_data) / np.sum(weights_data)
    log_var = np.sum(weights_data * (log_values - log_mean)**2) / np.sum(weights_data)
    log_std = np.sqrt(log_var)
    lognorm_params = (log_std, 0, np.exp(log_mean))
    
    mean_val = np.sum(x_shifted * weights_data) / np.sum(weights_data)
    var_val = np.sum(weights_data * (x_shifted - mean_val)**2) / np.sum(weights_data)
    a0 = mean_val**2 / var_val if var_val > 0 else 1.0
    scale0 = var_val / mean_val if mean_val > 0 else 1.0
    
    gamma1_params = (a0, 0, scale0)
    gamma2_params = (a0 * 0.5, 0, scale0 * 1.5)
    
    weights = np.array([1/3, 1/3, 1/3])
    
    for iteration in range(max_iter):
        gamma1_pdf = stats.gamma.pdf(x_shifted, *gamma1_params)
        gamma2_pdf = stats.gamma.pdf(x_shifted, *gamma2_params)
        lognorm_pdf = stats.lognorm.pdf(x_shifted, *lognorm_params)
        
        gamma1_weighted = weights[0] * gamma1_pdf
        gamma2_weighted = weights[1] * gamma2_pdf
        lognorm_weighted = weights[2] * lognorm_pdf
        total = gamma1_weighted + gamma2_weighted + lognorm_weighted
        total[total == 0] = 1e-10
        
        resp_gamma1 = gamma1_weighted / total
        resp_gamma2 = gamma2_weighted / total
        resp_lognorm = lognorm_weighted / total
        
        new_weights = np.array([
            np.sum(resp_gamma1 * weights_data) / np.sum(weights_data),
            np.sum(resp_gamma2 * weights_data) / np.sum(weights_data),
            np.sum(resp_lognorm * weights_data) / np.sum(weights_data)
        ])
        
        weighted_values_g1 = resp_gamma1 * weights_data
        if np.sum(weighted_values_g1) > 0:
            new_mean = np.sum(x_shifted * weighted_values_g1) / np.sum(weighted_values_g1)
            new_var = np.sum(weighted_values_g1 * (x_shifted - new_mean)**2) / np.sum(weighted_values_g1)
            new_a = new_mean**2 / new_var if new_var > 0 else 1.0
            new_scale = new_var / new_mean if new_mean > 0 else 1.0
            gamma1_params = (new_a, 0, new_scale)
        
        weighted_values_g2 = resp_gamma2 * weights_data
        if np.sum(weighted_values_g2) > 0:
            new_mean = np.sum(x_shifted * weighted_values_g2) / np.sum(weighted_values_g2)
            new_var = np.sum(weighted_values_g2 * (x_shifted - new_mean)**2) / np.sum(weighted_values_g2)
            new_a = new_mean**2 / new_var if new_var > 0 else 1.0
            new_scale = new_var / new_mean if new_mean > 0 else 1.0
            gamma2_params = (new_a, 0, new_scale)
        
        weighted_log_values = resp_lognorm * weights_data
        if np.sum(weighted_log_values) > 0:
            new_log_mean = np.sum(log_values * weighted_log_values) / np.sum(weighted_log_values)
            new_log_var = np.sum(weighted_log_values * (log_values - new_log_mean)**2) / np.sum(weighted_log_values)
            new_log_std = np.sqrt(new_log_var)
            lognorm_params = (new_log_std, 0, np.exp(new_log_mean))
        
        weight_diff = np.abs(new_weights - weights).max()
        weights = new_weights
        
        if weight_diff < tol:
            break
    
    return {
        'weights': weights,
        'gamma1_params': gamma1_params,
        'gamma2_params': gamma2_params,
        'lognorm_params': lognorm_params,
        'iterations': iteration + 1,
        'loc': loc
    }

def plot_mixture_fit_3dist(counts, mixture_result, thresholds, output_path):
    """
    Plot histogram + 3-component EM fit (Gamma + Gamma + Lognormal) — Fig4
    额外叠加两条对比竖线：
      - BIN 前置统计阈值「均值+标准差」(thresholds['mean_plus_std'], 本配置=均值+3σ)
      - 由三分布二阶导推导的相对曲率衰减阈值 λ* (曲率衰减) 交点 xc3
    说明：本图纵轴为「Page Count」(计数)，与分量 PDF 密度属不同量纲；
          因此仅以「竖直线 + x 坐标标注」标记阈值，避免把密度量纲误叠加到计数坐标
          造成拟合阈值标记错误。x 轴统一为原始 Error Bit Count 数据坐标，
          loc 偏移仅在分量 PDF 求值内部使用，不影响竖线位置。
    """
    if mixture_result is None:
        return
    
    left_bound, right_bound = get_auto_bounds(counts)
    max_bit = right_bound
    x = list(range(max_bit + 1))
    counts_plot = np.pad(counts[:max_bit + 1], (0, max_bit + 1 - len(counts[:max_bit + 1])), mode='constant')
    total_fit = np.sum(counts_plot[1:])
    
    fig, (summary_ax, ax) = plt.subplots(2, 1, figsize=(14, 11), dpi=100,
                                          gridspec_kw={'height_ratios': [1.5, 6]})
    
    summary_ax.axis('off')
    
    weights = mixture_result['weights']
    gamma1_params = mixture_result['gamma1_params']
    gamma2_params = mixture_result['gamma2_params']
    lognorm_params = mixture_result['lognorm_params']
    loc = mixture_result['loc']
    
    # ---- 对比阈值：BIN 前置统计 & 三分布二阶导 λ* & 一阶导 λ* ----
    mean_plus_std = thresholds['mean_plus_std']   # 本配置 = 均值 + 3σ
    xc3 = _compute_deriv_threshold(mixture_result, 3, right_bound)
    xc1_3 = _compute_deriv1_threshold(mixture_result, 3, right_bound)
    
    summary_text = f"""
EM Mixture Distribution Fit (Gamma + Gamma + Lognormal) — 3-dist
─────────────────────────────────────────────────────────────
Weights:
  Gamma 1:     {weights[0]:.4f}
  Gamma 2:     {weights[1]:.4f}
  Lognormal:   {weights[2]:.4f}

Gamma 1 Distribution:
  shape ({chr(945)}): {gamma1_params[0]:.4f}
  scale ({chr(952)}): {gamma1_params[2]:.4f}

Gamma 2 Distribution:
  shape ({chr(945)}): {gamma2_params[0]:.4f}
  scale ({chr(952)}): {gamma2_params[2]:.4f}

Lognormal Distribution:
  shape ({chr(963)}): {lognorm_params[0]:.4f}
  scale (e^{chr(956)}): {lognorm_params[2]:.4f}

Location Shift:
  loc = {loc:.4f}

Threshold Comparison (对比阈值):
  BIN 均值+标准差 (mean+std): {mean_plus_std:.4f}
""" + (f"  三分布二阶导 λ*阈值 (曲率衰减): {xc3:.4f}\n" if xc3 is not None
       else "  三分布二阶导 λ*阈值: N/A\n") + (f"  三分布一阶导 λ*阈值 (斜率衰减): {xc1_3:.4f}\n" if xc1_3 is not None
       else "  三分布一阶导 λ*阈值: N/A\n") + f"""
Converged in {mixture_result['iterations']} iterations
"""
    
    summary_ax.text(0.02, 0.95, summary_text, ha='left', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    ax.bar(x, counts_plot, color='#1f77b4', edgecolor='#1565c0', alpha=0.5, width=1.0, label='Histogram')
    
    x_plot = np.linspace(max(0.1, loc), right_bound, 500)
    x_plot_shifted = x_plot - loc
    bin_width = 1.0
    
    gamma1_pdf = stats.gamma.pdf(x_plot_shifted, *gamma1_params)
    gamma1_fit = gamma1_pdf * total_fit * bin_width * weights[0]
    ax.plot(x_plot, gamma1_fit, color='#2ca02c', linewidth=2, linestyle='--',
            label=f'Gamma 1 (w={weights[0]:.2f})')
    
    gamma2_pdf = stats.gamma.pdf(x_plot_shifted, *gamma2_params)
    gamma2_fit = gamma2_pdf * total_fit * bin_width * weights[1]
    ax.plot(x_plot, gamma2_fit, color='#98df8a', linewidth=2, linestyle=':',
            label=f'Gamma 2 (w={weights[1]:.2f})')
    
    lognorm_pdf = stats.lognorm.pdf(x_plot_shifted, *lognorm_params)
    lognorm_fit = lognorm_pdf * total_fit * bin_width * weights[2]
    ax.plot(x_plot, lognorm_fit, color='#ff7f0e', linewidth=2, linestyle='-',
            label=f'Lognormal (w={weights[2]:.2f})')
    
    mixture_fit = gamma1_fit + gamma2_fit + lognorm_fit
    ax.plot(x_plot, mixture_fit, color='#9467bd', linewidth=3, linestyle='-.',
            label='Mixture (Gamma + Gamma + Lognormal)')
    
    # ---- 对比竖线：统计阈值 (均值+标准差) 与 三分布二阶导 λ* 阈值 ----
    # 注意：本图纵轴为「Page Count」(计数)，与 PDF 密度不同量纲；
    # 故仅用竖直线 + x 坐标标注，避免把密度量纲误叠加到计数坐标。
    # x 轴为原始 Error Bit Count，loc 偏移只用于分量 PDF 求值，不影响竖线位置。
    ax.axvline(mean_plus_std, color='#2ca02c', linestyle='--', linewidth=2.0,
               label=f'均值+标准差 = {mean_plus_std:.2f}')

    if xc3 is not None:
        ax.axvline(xc3, color='#E67E22', linestyle=(0, (3, 1)), linewidth=2.4,
                   label=f'三分布二阶导 λ*阈值 x≈{xc3:.2f}')

    if xc1_3 is not None:
        ax.axvline(xc1_3, color='#FFA500', linestyle=(0, (5, 1, 1, 1)), linewidth=2.2,
                   label=f'三分布一阶导 λ*阈值 x≈{xc1_3:.2f}')

    # 顶部标注（axis-fraction 坐标，避免受计数纵轴放大影响）
    _ytop = ax.get_ylim()[1]
    if xc3 is not None:
        ax.annotate(f'λ*阈值 (曲率衰减)\nx≈{xc3:.2f}',
                    xy=(xc3, 0.95 * _ytop),
                    xytext=(xc3 - 22, 0.90 * _ytop),
                    ha='center', color='#E67E22', fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#E67E22'))
        _gap = xc3 - mean_plus_std
        _info_text = (f"三分布二阶导 λ* 阈值 x≈{xc3:.2f}\n"
                      f"统计阈值 均值+标准差 = {mean_plus_std:.2f}\n"
                      f"差值 (λ*-统计) = {_gap:+.2f}")
        if xc1_3 is not None:
            _info_text += f"\n三分布一阶导 λ* 阈值 x≈{xc1_3:.2f}"
        ax.text(0.02, 0.97, _info_text,
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))

    if xc1_3 is not None:
        ax.annotate(f'一阶导 λ*阈值\n(斜率衰减) x≈{xc1_3:.2f}',
                    xy=(xc1_3, 0.95 * _ytop),
                    xytext=(xc1_3 + 8, 0.70 * _ytop),
                    ha='left', color='#FFA500', fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#FFA500'))

    ax.annotate(f'均值+标准差\n={mean_plus_std:.2f}',
                xy=(mean_plus_std, 0.95 * _ytop),
                xytext=(mean_plus_std + 6, 0.80 * _ytop),
                ha='left', color='#2ca02c', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ca02c'))

    ax.set_xlabel('Error Bit Count', fontsize=12)
    ax.set_ylabel('Page Count', fontsize=12)
    ax.set_title('EM Mixture Fit [3-dist: Gamma+Gamma+Lognormal]\n二阶导/一阶导 λ* 阈值 vs 均值+标准差 对比', fontsize=14, pad=20)
    ax.set_xlim(-5, right_bound + 5)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Fig4: {output_path}")

# ============================================================
# EM mixture 2-dist (Lognormal + Gamma) — Fig5: mixture_fit2.png (Control)
# ============================================================
def fit_mixture_em_2dist(counts, max_iter=100, tol=1e-6):
    """
    EM algorithm for 2-component mixture: Lognormal + Gamma (Control)
    Initial weights: [0.5, 0.5]
    """
    max_bit = min(len(counts) - 1, 1000)
    counts = counts[:max_bit + 1]
    
    start_with_zero = counts[0] > 0
    
    first_non_zero = None
    for value, count in enumerate(counts):
        if count > 0:
            first_non_zero = value
            break
    
    values = []
    weights_data = []
    for value, count in enumerate(counts):
        if value > 0 and count > 0:
            values.append(value)
            weights_data.append(count)
    
    if len(values) < 10:
        return None
    
    values = np.array(values)
    weights_data = np.array(weights_data)
    
    if start_with_zero:
        loc = 0.0
        x_shifted = values
    else:
        loc = first_non_zero - 1.0
        x_shifted = values - loc
    
    log_values = np.log(x_shifted)
    log_mean = np.sum(log_values * weights_data) / np.sum(weights_data)
    log_var = np.sum(weights_data * (log_values - log_mean)**2) / np.sum(weights_data)
    log_std = np.sqrt(log_var)
    lognorm_params = (log_std, 0, np.exp(log_mean))
    
    mean_val = np.sum(x_shifted * weights_data) / np.sum(weights_data)
    var_val = np.sum(weights_data * (x_shifted - mean_val)**2) / np.sum(weights_data)
    a0 = mean_val**2 / var_val if var_val > 0 else 1.0
    scale0 = var_val / mean_val if mean_val > 0 else 1.0
    gamma_params = (a0, 0, scale0)
    
    weights = np.array([0.50, 0.50])
    
    for iteration in range(max_iter):
        lognorm_pdf = stats.lognorm.pdf(x_shifted, *lognorm_params)
        gamma_pdf = stats.gamma.pdf(x_shifted, *gamma_params)
        
        lognorm_weighted = weights[0] * lognorm_pdf
        gamma_weighted = weights[1] * gamma_pdf
        total = lognorm_weighted + gamma_weighted
        total[total == 0] = 1e-10
        
        resp_lognorm = lognorm_weighted / total
        resp_gamma = gamma_weighted / total
        
        new_weights = np.array([
            np.sum(resp_lognorm * weights_data) / np.sum(weights_data),
            np.sum(resp_gamma * weights_data) / np.sum(weights_data)
        ])
        
        weighted_log_values = resp_lognorm * weights_data
        if np.sum(weighted_log_values) > 0:
            new_log_mean = np.sum(log_values * weighted_log_values) / np.sum(weighted_log_values)
            new_log_var = np.sum(weighted_log_values * (log_values - new_log_mean)**2) / np.sum(weighted_log_values)
            new_log_std = np.sqrt(new_log_var)
            lognorm_params = (new_log_std, 0, np.exp(new_log_mean))
        
        weighted_values_g = resp_gamma * weights_data
        if np.sum(weighted_values_g) > 0:
            new_mean = np.sum(x_shifted * weighted_values_g) / np.sum(weighted_values_g)
            new_var = np.sum(weighted_values_g * (x_shifted - new_mean)**2) / np.sum(weighted_values_g)
            new_a = new_mean**2 / new_var if new_var > 0 else 1.0
            new_scale = new_var / new_mean if new_mean > 0 else 1.0
            gamma_params = (new_a, 0, new_scale)
        
        weight_diff = np.abs(new_weights - weights).max()
        weights = new_weights
        
        if weight_diff < tol:
            break
    
    return {
        'weights': weights,
        'lognorm_params': lognorm_params,
        'gamma_params': gamma_params,
        'iterations': iteration + 1,
        'loc': loc
    }

def plot_mixture_fit_2dist(counts, mixture_result, thresholds, output_path):
    """
    Plot histogram + 2-component EM fit (Lognormal + Gamma) — Fig5 (Control)
    额外叠加两条对比竖线（与 mixture_fit3.png 一致）：
      - BIN 前置统计阈值「均值+标准差」(thresholds['mean_plus_std'], 本配置=均值+3σ)
      - 由二分布二阶导推导的相对曲率衰减阈值 λ* (曲率衰减) 交点 xc2
    说明同 mixture_fit3.png：本图纵轴为「Page Count」(计数)，与 PDF 密度不同量纲；
          仅以竖直线 + x 坐标标注标记阈值，避免误标。
    """
    if mixture_result is None:
        return
    
    left_bound, right_bound = get_auto_bounds(counts)
    max_bit = right_bound
    x = list(range(max_bit + 1))
    counts_plot = np.pad(counts[:max_bit + 1], (0, max_bit + 1 - len(counts[:max_bit + 1])), mode='constant')
    
    fig, (summary_ax, ax) = plt.subplots(2, 1, figsize=(14, 11), dpi=100,
                                          gridspec_kw={'height_ratios': [1.5, 6]})
    
    summary_ax.axis('off')
    
    weights = mixture_result['weights']
    lognorm_params = mixture_result['lognorm_params']
    gamma_params = mixture_result['gamma_params']
    loc = mixture_result['loc']
    
    # ---- 对比阈值：BIN 前置统计 & 二分布二阶导 λ* & 一阶导 λ* ----
    mean_plus_std = thresholds['mean_plus_std']   # 本配置 = 均值 + 3σ
    xc2 = _compute_deriv_threshold(mixture_result, 2, right_bound)
    xc1_2 = _compute_deriv1_threshold(mixture_result, 2, right_bound)
    
    summary_text = f"""
EM Mixture Distribution Fit (Lognormal + Gamma) — 2-dist Control
─────────────────────────────────────────────────────────────
Initial Weights: [0.5, 0.5]
Final Weights:
  Lognormal: {weights[0]:.4f}
  Gamma:     {weights[1]:.4f}

Lognormal Distribution:
  shape ({chr(963)}): {lognorm_params[0]:.4f}
  scale (e^{chr(956)}): {lognorm_params[2]:.4f}

Gamma Distribution:
  shape ({chr(945)}): {gamma_params[0]:.4f}
  scale ({chr(952)}): {gamma_params[2]:.4f}

Location Shift:
  loc = {loc:.4f}

Threshold Comparison (对比阈值):
  BIN 均值+标准差 (mean+std): {mean_plus_std:.4f}
""" + (f"  二分布二阶导 λ*阈值 (曲率衰减): {xc2:.4f}\n" if xc2 is not None
       else "  二分布二阶导 λ*阈值: N/A\n") + (f"  二分布一阶导 λ*阈值 (斜率衰减): {xc1_2:.4f}\n" if xc1_2 is not None
       else "  二分布一阶导 λ*阈值: N/A\n") + f"""
Converged in {mixture_result['iterations']} iterations
"""
    
    summary_ax.text(0.02, 0.95, summary_text, ha='left', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    ax.bar(x, counts_plot, color='#1f77b4', edgecolor='#1565c0', alpha=0.5, width=1.0, label='Histogram')
    
    x_plot = np.linspace(max(0.1, loc), right_bound, 500)
    x_plot_shifted = x_plot - loc
    bin_width = 1.0
    total_fit = np.sum(counts_plot[1:])
    
    lognorm_pdf = stats.lognorm.pdf(x_plot_shifted, *lognorm_params)
    lognorm_fit = lognorm_pdf * total_fit * bin_width * weights[0]
    ax.plot(x_plot, lognorm_fit, color='#ff7f0e', linewidth=2, linestyle='-',
            label=f'Lognormal (w={weights[0]:.2f})')
    
    gamma_pdf = stats.gamma.pdf(x_plot_shifted, *gamma_params)
    gamma_fit = gamma_pdf * total_fit * bin_width * weights[1]
    ax.plot(x_plot, gamma_fit, color='#2ca02c', linewidth=2, linestyle='--',
            label=f'Gamma (w={weights[1]:.2f})')
    
    mixture_fit = lognorm_fit + gamma_fit
    ax.plot(x_plot, mixture_fit, color='#9467bd', linewidth=3, linestyle='-.',
            label='Mixture (Lognormal + Gamma)')
    
    # ---- 对比竖线：统计阈值 (均值+标准差) 与 二分布二阶导 λ* 阈值 ----
    # 与 mixture_fit3.png 一致：仅竖直线 + x 坐标标注，避免密度量纲误叠加到计数坐标
    ax.axvline(mean_plus_std, color='#2ca02c', linestyle='--', linewidth=2.0,
               label=f'均值+标准差 = {mean_plus_std:.2f}')

    if xc2 is not None:
        ax.axvline(xc2, color='#1F4E79', linestyle=(0, (3, 1)), linewidth=2.4,
                   label=f'二分布二阶导 λ*阈值 x≈{xc2:.2f}')

    if xc1_2 is not None:
        ax.axvline(xc1_2, color='#2E86C1', linestyle=(0, (5, 1, 1, 1)), linewidth=2.2,
                   label=f'二分布一阶导 λ*阈值 x≈{xc1_2:.2f}')

    _ytop = ax.get_ylim()[1]
    if xc2 is not None:
        ax.annotate(f'λ*阈值 (曲率衰减)\nx≈{xc2:.2f}',
                    xy=(xc2, 0.95 * _ytop),
                    xytext=(xc2 - 22, 0.90 * _ytop),
                    ha='center', color='#1F4E79', fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#1F4E79'))
        _gap = xc2 - mean_plus_std
        _info_text = (f"二分布二阶导 λ* 阈值 x≈{xc2:.2f}\n"
                      f"统计阈值 均值+标准差 = {mean_plus_std:.2f}\n"
                      f"差值 (λ*-统计) = {_gap:+.2f}")
        if xc1_2 is not None:
            _info_text += f"\n二分布一阶导 λ* 阈值 x≈{xc1_2:.2f}"
        ax.text(0.02, 0.97, _info_text,
                transform=ax.transAxes, va='top', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.85))

    if xc1_2 is not None:
        ax.annotate(f'一阶导 λ*阈值\n(斜率衰减) x≈{xc1_2:.2f}',
                    xy=(xc1_2, 0.95 * _ytop),
                    xytext=(xc1_2 + 8, 0.70 * _ytop),
                    ha='left', color='#2E86C1', fontsize=9, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='#2E86C1'))

    ax.annotate(f'均值+标准差\n={mean_plus_std:.2f}',
                xy=(mean_plus_std, 0.95 * _ytop),
                xytext=(mean_plus_std + 6, 0.80 * _ytop),
                ha='left', color='#2ca02c', fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='#2ca02c'))

    ax.set_xlabel('Error Bit Count', fontsize=12)
    ax.set_ylabel('Page Count', fontsize=12)
    ax.set_title('EM Mixture Fit [2-dist: Lognormal+Gamma] - Control\n二阶导/一阶导 λ* 阈值 vs 均值+标准差 对比', fontsize=14, pad=20)
    ax.set_xlim(-5, right_bound + 5)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Fig5: {output_path}")

# ============================================================
# 新参数导数分析图 (参考 plot_mixture_newparams.py)
# 依据双分布 / 三分布 EM 拟合结果，绘制 f、f'、f'' 及关键位置
# 仅绘制「均值 + 3σ」竖线（均值本身不画）；
# 均值与标准差取自 BIN 文件前置计算得到的统计结果 (stats)
# ============================================================

# ---------- 分量 PDF 与解析导数 (避免数值微分噪声) ----------
def _gamma_pdf(u, a, scale):
    return stats.gamma.pdf(u, a, loc=0, scale=scale)

def _gamma_d1(u, a, scale):
    out = np.zeros_like(u, dtype=float)
    m = u > 0
    um = u[m]
    f = stats.gamma.pdf(um, a, loc=0, scale=scale)
    out[m] = f * ((a - 1.0) / um - 1.0 / scale)
    return out

def _gamma_d2(u, a, scale):
    out = np.zeros_like(u, dtype=float)
    m = u > 0
    um = u[m]
    f = stats.gamma.pdf(um, a, loc=0, scale=scale)
    h = (a - 1.0) / um - 1.0 / scale
    out[m] = f * (h**2 - (a - 1.0) / (um**2))
    return out

def _logn_pdf(u, s, scale):
    return stats.lognorm.pdf(u, s, loc=0, scale=scale)

def _logn_d1(u, s, scale):
    out = np.zeros_like(u, dtype=float)
    m = u > 0
    um = u[m]
    f = stats.lognorm.pdf(um, s, loc=0, scale=scale)
    mu = np.log(scale)
    a = 1.0 + (np.log(um) - mu) / (s**2)
    out[m] = f * (-1.0 / um) * a
    return out

def _logn_d2(u, s, scale):
    out = np.zeros_like(u, dtype=float)
    m = u > 0
    um = u[m]
    f = stats.lognorm.pdf(um, s, loc=0, scale=scale)
    mu = np.log(scale)
    a = 1.0 + (np.log(um) - mu) / (s**2)
    out[m] = f * (1.0 / (um**2)) * (a**2 + a - 1.0 / (s**2))
    return out

def _mixture_components(mixture_result, n_dist):
    """返回分量列表 [(kind, params, weight, loc)]。"""
    w = mixture_result['weights']
    loc = mixture_result['loc']
    if n_dist == 3:
        return [
            ('gamma', mixture_result['gamma1_params'], w[0], loc),
            ('gamma', mixture_result['gamma2_params'], w[1], loc),
            ('lognorm', mixture_result['lognorm_params'], w[2], loc),
        ]
    else:  # 2-dist: Lognormal + Gamma (Control)
        return [
            ('lognorm', mixture_result['lognorm_params'], w[0], loc),
            ('gamma', mixture_result['gamma_params'], w[1], loc),
        ]

def _all_peaks(y):
    pos = find_peaks(y, prominence=0.01 * np.max(np.abs(y)))[0]
    neg = find_peaks(-y, prominence=0.01 * np.max(np.abs(y)))[0]
    return np.sort(np.unique(np.concatenate([pos, neg])))

def _rightmost_positive_peak(fpp, x, prominence_frac=0.003):
    """返回 f'' 最右侧正峰(右极值) 的索引；找不到返回 None。
    用较低的 prominence 以捕获真正的「最右侧极值点」，
    避免因 prominence 过高把真正最右的极值点滤掉，导致阈值偏左。"""
    def _peaks(pf):
        pk, _ = find_peaks(fpp, prominence=pf * np.max(np.abs(fpp)))
        return [i for i in pk if x[i] > 0 and fpp[i] > 0]
    rp = _peaks(prominence_frac)
    if not rp:
        rp = _peaks(0.0)          # 退化：取全部正峰
    return rp[-1] if rp else None

def _curvature_decay(fpp, x, lam, prominence_frac=0.003):
    """二阶导右极值 M 之后，衰减到 λ·M 的第一个 x。
    取 f'' 最右侧的正峰作为右极值 M（用较低 prominence 捕获真正最右的极值点）。"""
    iM = _rightmost_positive_peak(fpp, x, prominence_frac)
    if iM is None:
        return None, None, None
    xM, M = x[iM], fpp[iM]
    target = lam * M
    idx = np.where((x > xM) & (fpp <= target))[0]
    if len(idx) == 0:
        return None, xM, M
    i = idx[0]
    xa, xb = x[i - 1], x[i]
    ya, yb = fpp[i - 1], fpp[i]
    xc = xa + (target - ya) * (xb - xa) / (yb - ya) if yb != ya else xb
    return xc, xM, M

def _compute_deriv_threshold(mixture_result, n_dist, right_bound):
    """由混合分布二阶导的相对曲率衰减 (λ*≈10%, 曲率衰减) 求阈值 x（数据坐标）。"""
    components = _mixture_components(mixture_result, n_dist)
    loc = mixture_result['loc']
    x = np.linspace(max(0.1, loc), right_bound, 14000)
    u = x - loc
    fpp = np.zeros_like(x)
    for kind, params, w, loc_c in components:
        if kind == 'gamma':
            a, _, scale = params
            fpp += w * _gamma_d2(u, a, scale)
        else:
            s, _, scale = params
            fpp += w * _logn_d2(u, s, scale)
    LAMBDA_3SIGMA = 0.10
    xc3, _, _ = _curvature_decay(fpp, x, LAMBDA_3SIGMA)
    return xc3

# ---------- 一阶导斜率衰减阈值 (与二阶导曲率衰减对偶) ----------
def _rightmost_negative_peak(fp, x, prominence_frac=0.003):
    """返回 f' 最右侧负峰(最负极值) 的索引；找不到返回 None。
    与 _rightmost_positive_peak 对偶：在一阶导的下降尾段，
    最右侧的负极值代表「斜率最陡的下降点」。"""
    def _peaks(pf):
        pk, _ = find_peaks(-fp, prominence=pf * np.max(np.abs(fp)))
        return [i for i in pk if x[i] > 0 and fp[i] < 0]
    rp = _peaks(prominence_frac)
    if not rp:
        rp = _peaks(0.0)          # 退化：取全部负峰
    return rp[-1] if rp else None

def _slope_decay(fp, x, lam, prominence_frac=0.003):
    """一阶导右极值 N(负谷) 之后，回升到 λ·N 的第一个 x。
    与 _curvature_decay 对偶：N 为负，λ·N 也是负但更接近 0，
    表示斜率绝对值衰减到峰值的 λ 倍。"""
    iN = _rightmost_negative_peak(fp, x, prominence_frac)
    if iN is None:
        return None, None, None
    xN, N = x[iN], fp[iN]
    target = lam * N               # N<0, target 也是负但更接近 0
    idx = np.where((x > xN) & (fp >= target))[0]
    if len(idx) == 0:
        return None, xN, N
    i = idx[0]
    xa, xb = x[i - 1], x[i]
    ya, yb = fp[i - 1], fp[i]
    xc = xa + (target - ya) * (xb - xa) / (yb - ya) if yb != ya else xb
    return xc, xN, N

def _compute_deriv1_threshold(mixture_result, n_dist, right_bound):
    """由混合分布一阶导的相对斜率衰减 (λ*≈5%, 斜率衰减) 求阈值 x（数据坐标）。"""
    components = _mixture_components(mixture_result, n_dist)
    loc = mixture_result['loc']
    x = np.linspace(max(0.1, loc), right_bound, 14000)
    u = x - loc
    fp = np.zeros_like(x)
    for kind, params, w, loc_c in components:
        if kind == 'gamma':
            a, _, scale = params
            fp += w * _gamma_d1(u, a, scale)
        else:
            s, _, scale = params
            fp += w * _logn_d1(u, s, scale)
    LAMBDA = 0.05
    xc1, _, _ = _slope_decay(fp, x, LAMBDA)
    return xc1

def plot_mixture_newparams_annotated(counts, mixture_result, n_dist, stats, output_path):
    """
    参考 plot_mixture_newparams.py 的绘图风格：
    绘制混合分布 f(x)、一阶导 f'(x)、二阶导 f''(x) 及关键位置。
    仅绘制「均值 + 3σ」竖线（均值本身不画）。
    均值与标准差采用 BIN 文件前置计算得到的统计 (stats['mean'], stats['std'])。
    """
    if mixture_result is None:
        return
    
    components = _mixture_components(mixture_result, n_dist)
    loc = mixture_result['loc']
    
    # ---- 数值来源：BIN 文件前置统计 ----
    mean_b = stats['mean']
    std_b = stats['std']
    thr_b = mean_b + 3.0 * std_b   # 仅画 均值 + 3σ
    
    left_bound, right_bound = get_auto_bounds(counts)
    
    x = np.linspace(max(0.1, loc), right_bound, 14000)
    u = x - loc
    
    f = np.zeros_like(x)
    fp = np.zeros_like(x)
    fpp = np.zeros_like(x)
    for kind, params, w, loc_c in components:
        if kind == 'gamma':
            a, _, scale = params
            cf = w * _gamma_pdf(u, a, scale)
            cfp = w * _gamma_d1(u, a, scale)
            cfpp = w * _gamma_d2(u, a, scale)
        else:
            s, _, scale = params
            cf = w * _logn_pdf(u, s, scale)
            cfp = w * _logn_d1(u, s, scale)
            cfpp = w * _logn_d2(u, s, scale)
        f += cf
        fp += cfp
        fpp += cfpp
    
    # ---- 关键位置检测 ----
    pk_f = find_peaks(f, prominence=0.01 * np.max(f))[0]
    peaks = [i for i in pk_f if fpp[i] < 0]
    
    infl_idx = _all_peaks(fp)
    infl_idx = [i for i in infl_idx if np.abs(fpp[i]) < 0.02 * np.max(np.abs(fpp)) + 1e-6]
    
    fpp_ext_idx = _all_peaks(fpp)
    
    # ---- 相对曲率衰减阈值 (比例 λ*) ----
    LAMBDA_3SIGMA = 0.10
    iM_rp = _rightmost_positive_peak(fpp, x)
    xM = M = None
    if iM_rp is not None:
        xM, M = x[iM_rp], fpp[iM_rp]
    
    xc3, _, _ = _curvature_decay(fpp, x, LAMBDA_3SIGMA)
    
    # ---- 一阶导斜率衰减阈值 (比例 λ*, 与二阶导曲率衰减对偶) ----
    LAMBDA_SLOPE = 0.05
    iN_rp = _rightmost_negative_peak(fp, x)
    xN = N_val = None
    if iN_rp is not None:
        xN, N_val = x[iN_rp], fp[iN_rp]
    xc1, _, _ = _slope_decay(fp, x, LAMBDA_SLOPE)
    
    # ---- 竖线配色：3-dist 橙色系 / 2-dist 蓝色系 ----
    if n_dist == 3:
        _clr_d2 = "#E67E22"   # 3-dist 二阶导: 深橙
        _clr_d1 = "#FFA500"   # 3-dist 一阶导: 橙
    else:
        _clr_d2 = "#1F4E79"   # 2-dist 二阶导: 深蓝
        _clr_d1 = "#2E86C1"   # 2-dist 一阶导: 中蓝
    _ls_dense_dash = (0, (3, 1))           # 密集虚线
    _ls_dense_dashdot = (0, (5, 1, 1, 1)) # 密集点横线
    
    # ---- 绘图右边界：聚焦到 均值+3σ 附近 ----
    xlim_right = thr_b + 0.4 * std_b
    
    c_f, c_fp, c_fpp = "#1f77b4", "#ff7f0e", "#2ca02c"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(19, 7.2), layout="constrained")
    
    # ===== 左图：f, f', f'' + 关键竖线 =====
    ax1.plot(x, f,   label=r"和分布 $f(x)$", color=c_f, lw=2.6)
    ax1.plot(x, fp,  label=r"一阶导 $f'(x)$", color=c_fp, lw=2.0, ls="--")
    ax1.plot(x, fpp, label=r"二阶导 $f''(x)$", color=c_fpp, lw=2.0, ls="-.")
    
    ax1.axhline(0, color="black", lw=0.8)
    ax1.axvline(0, color="black", lw=1.0)
    
    # 仅绘制 均值 + 3σ (均值本身不画)
    ax1.axvline(thr_b, color="red", ls="--", lw=1.4)
    ax1.annotate("均值+3σ={:.2f}".format(thr_b), xy=(thr_b, 0), xytext=(0.80, 0.93),
                 textcoords="axes fraction", ha="center", color="red", fontsize=9,
                 fontweight="bold")
    
    # 相对曲率衰减阈值 (比例 λ*) — 二阶导竖线 (3-dist 橙/2-dist 蓝)
    if xc3 is not None and xc3 <= xlim_right:
        _y = 0.88 * ax1.get_ylim()[1]
        ax1.axvline(xc3, color=_clr_d2, ls=_ls_dense_dash, lw=2.0)
        ax1.annotate("λ*≈{:.1f}%  (曲率衰减)\nx≈{:.2f}".format(LAMBDA_3SIGMA * 100, xc3),
                     xy=(xc3, 0), xytext=(xc3 - 5.0, _y),
                     ha="center", color=_clr_d2, fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=_clr_d2))
    
    # 一阶导斜率衰减阈值 (比例 λ*) — 一阶导竖线 (3-dist 橙/2-dist 蓝)
    if xc1 is not None and xc1 <= xlim_right:
        _y1 = 0.75 * ax1.get_ylim()[1]
        ax1.axvline(xc1, color=_clr_d1, ls=_ls_dense_dashdot, lw=2.0)
        ax1.annotate("一阶导 λ*≈{:.1f}%  (斜率衰减)\nx≈{:.2f}".format(LAMBDA_SLOPE * 100, xc1),
                     xy=(xc1, 0), xytext=(xc1 + 6.0, _y1),
                     ha="center", color=_clr_d1, fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=_clr_d1))
    
    # 标记一阶导右极值(负谷) N
    if xN is not None and xN <= xlim_right:
        ax1.plot(xN, N_val, "D", color=_clr_d1, ms=8)
    
    summ_l = "{} 分布 (EM 拟合) 阈值\n".format("三" if n_dist == 3 else "双")
    if xc3 is not None:
        summ_l += "★ λ*≈{:.1f}% (曲率衰减): x={:.2f}  (vs 均值+3σ={:.2f})\n".format(LAMBDA_3SIGMA * 100, xc3, thr_b)
    if xN is not None:
        summ_l += "◆ f'右极值 N: x={:.2f}\n".format(xN)
    if xc1 is not None:
        summ_l += "★ 一阶导 λ*≈{:.1f}% (斜率衰减): x={:.2f}".format(LAMBDA_SLOPE * 100, xc1)
    ax1.text(0.02, 0.97, summ_l, transform=ax1.transAxes, va="top", fontsize=9,
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    
    ax1.set_xlim(0, xlim_right)
    _ybot = min(fpp.min(), fp.min())
    _ytop = max(f.max(), fpp.max(), fp.max())
    ax1.set_ylim(1.15 * _ybot, 1.12 * _ytop)
    ax1.set_xlabel("x  (Error Bit Count)")
    ax1.set_ylabel("函数值")
    ax1.set_title("加权和分布 f(x) 及其一阶导、二阶导 + 衰减阈值")
    ax1.legend(loc="upper right", fontsize=8.5)
    ax1.grid(True, alpha=0.3)
    
    # ===== 右图：f'' 与 f' 特写 =====
    ax2.plot(x, fpp, color=c_fpp, lw=2.6, label=r"$f''(x)$")
    ax2.plot(x, fp,  color=c_fp,  lw=2.0, ls="--", label=r"$f'(x)$")
    ax2.axhline(0, color="black", lw=0.8)
    ax2.axvline(0, color="black", lw=0.8, alpha=0.6)
    
    # 仅绘制 均值 + 3σ
    ax2.axvline(thr_b, color="red", ls="--", lw=1.4)
    
    # 二阶导 λ·M 水平线 (曲率衰减交点水平线)
    if M is not None:
        ax2.axhline(LAMBDA_3SIGMA * M, color=_clr_d2, ls=":", lw=1.2, alpha=0.8,
                    label=r"$\lambda^*\!\cdot\!M$ ({:.0f}%)".format(LAMBDA_3SIGMA * 100))
    
    # 一阶导 λ·N 水平线 (斜率衰减交点水平线)
    if N_val is not None:
        ax2.axhline(LAMBDA_SLOPE * N_val, color=_clr_d1, ls=":", lw=1.2, alpha=0.8,
                    label=r"$\lambda^*\!\cdot\!N$ ({:.0f}%)".format(LAMBDA_SLOPE * 100))
    
    # 二阶导右极值 M 标记
    if xM is not None:
        ax2.plot(xM, M, "*", color=_clr_d2, ms=14)
        ax2.annotate("右极值 M\nx_M={:.2f}".format(xM), xy=(xM, M),
                     xytext=(xM - 7, M + 0.35 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])),
                     color=_clr_d2, fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=_clr_d2))
    
    # 一阶导右极值 N 标记
    if xN is not None:
        ax2.plot(xN, N_val, "D", color=_clr_d1, ms=10)
        ax2.annotate("右极值 N\nx_N={:.2f}".format(xN), xy=(xN, N_val),
                     xytext=(xN + 5, N_val - 0.3 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])),
                     color=_clr_d1, fontsize=9, fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color=_clr_d1))
    
    # 二阶导曲率衰减阈值竖线
    if xc3 is not None and M is not None:
        ylev = LAMBDA_3SIGMA * M
        if xc3 <= xlim_right:
            ax2.axvline(xc3, color=_clr_d2, ls=_ls_dense_dash, lw=2.0)
            ax2.annotate("λ*≈{:.1f}%  (曲率衰减)\nx≈{:.2f}".format(LAMBDA_3SIGMA * 100, xc3),
                         xy=(xc3, ylev), xytext=(xc3 - 12, 0.5 * (ax2.get_ylim()[1] - ax2.get_ylim()[0])),
                         color=_clr_d2, fontsize=9, fontweight="bold",
                         arrowprops=dict(arrowstyle="->", color=_clr_d2))
        else:
            ax2.axvline(xlim_right, color=_clr_d2, ls=_ls_dense_dash, lw=2.0)
            ax2.annotate("λ*≈{:.1f}% 阈值\nx≈{:.2f} (界外)".format(LAMBDA_3SIGMA * 100, xc3),
                         xy=(xlim_right, 0), xytext=(0.52, 0.42),
                         textcoords="axes fraction", ha="center", color=_clr_d2, fontsize=8.5,
                         fontweight="bold", arrowprops=dict(arrowstyle="->", color=_clr_d2))
    
    # 一阶导斜率衰减阈值竖线
    if xc1 is not None and xc1 <= xlim_right:
        ylev1 = LAMBDA_SLOPE * N_val if N_val is not None else 0
        ax2.axvline(xc1, color=_clr_d1, ls=_ls_dense_dashdot, lw=1.8)
        ax2.annotate("一阶导 λ*≈{:.1f}%\n(斜率衰减) x≈{:.2f}".format(LAMBDA_SLOPE * 100, xc1),
                     xy=(xc1, ylev1), xytext=(0.75, 0.58),
                     textcoords="axes fraction", ha="center", color=_clr_d1, fontsize=8.5,
                     fontweight="bold", arrowprops=dict(arrowstyle="->", color=_clr_d1))
    
    summ_r = "右图 (f'' 与 f' 特写, {}分布)\n".format("三" if n_dist == 3 else "双")
    if xM is not None:
        summ_r += "★ f''右极值 M: x={:.2f}\n".format(xM)
    if xc3 is not None:
        summ_r += "★ λ*≈{:.1f}%(曲率衰减) 阈值: x={:.2f}\n".format(LAMBDA_3SIGMA * 100, xc3)
    if xN is not None:
        summ_r += "◆ f'右极值 N: x={:.2f}\n".format(xN)
    if xc1 is not None:
        summ_r += "★ 一阶导 λ*≈{:.1f}%(斜率衰减) 阈值: x={:.2f}".format(LAMBDA_SLOPE * 100, xc1)
    ax2.text(0.02, 0.97, summ_r, transform=ax2.transAxes, va="top", fontsize=8.5,
             bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.85))
    
    ax2.set_xlim(0, xlim_right)
    _ybot_r = min(fpp.min(), fp.min())
    _ytop_r = max(fpp.max(), fp.max())
    ax2.set_ylim(1.15 * _ybot_r, 1.15 * _ytop_r)
    ax2.set_xlabel("x  (Error Bit Count)")
    ax2.set_ylabel(r"$f''(x)$ / $f'(x)$")
    ax2.set_title(r"$f''(x)$ 与 $f'(x)$ 特写：右极值 与 衰减阈值")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Fig: {output_path}")

def get_auto_bounds(counts):
    max_bit = min(len(counts) - 1, 1000)
    max_data = 0
    for value in range(max_bit, 0, -1):
        if counts[value] > 0:
            max_data = value
            break
    
    if max_data <= 250:
        right_bound = 250
    elif max_data <= 500:
        right_bound = 500
    else:
        right_bound = 1000
    
    return 0, right_bound

def plot_count_distribution(counts, thresholds, percentages, output_path, xc2=None, xc3=None, xc1_2=None, xc1_3=None, config=None):
    # 兼容：优先使用传入的 config，否则回退到模块全局 config
    if config is None:
        config = globals().get('config', {})
    left_bound, right_bound = get_auto_bounds(counts)
    max_bit = right_bound
    x = list(range(max_bit + 1))
    counts = np.pad(counts[:max_bit + 1], (0, max_bit + 1 - len(counts[:max_bit + 1])), mode='constant')
    
    fig, (summary_ax, ax) = plt.subplots(2, 1, figsize=(14, 11), dpi=100,
                                          gridspec_kw={'height_ratios': [1.5, 6]})
    
    summary_ax.axis('off')
    
    stats_text = f"""
Statistics Summary:
─────────────────────────────────────────────────────────────
Page Count: {sum(counts):,}

Mean Series:
  Mean: {thresholds['mean']:.4f} ({percentages['mean_pct']:.2f}% ≤)
  Mean + {config['std_multiplier']}xStd: {thresholds['mean_plus_std']:.4f} ({percentages['mean_plus_std_pct']:.2f}% ≤)
"""
    
    summary_ax.text(0.02, 0.95, stats_text, ha='left', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    ax.bar(x, counts, color='#1f77b4', edgecolor='#1565c0', alpha=0.7, width=1.0)
    
    # 统计阈值：均值+标准差（按需求去掉「均值」本身的划线）
    ax.axvline(x=thresholds['mean_plus_std'], color='#9467bd', linestyle='--', linewidth=2,
               label=f'Mean + {config["std_multiplier"]}xStd')
    # 两种分布(2-dist / 3-dist) 二阶导 λ* 阈值线 — 密集虚线
    if xc2 is not None:
        ax.axvline(x=xc2, color='#1F4E79', linestyle=(0, (3, 1)), linewidth=2.2,
                   label=f'2-dist 二阶导 λ* 阈值 x≈{xc2:.2f}')
    if xc3 is not None:
        ax.axvline(x=xc3, color='#E67E22', linestyle=(0, (3, 1)), linewidth=2.4,
                   label=f'3-dist 二阶导 λ* 阈值 x≈{xc3:.2f}')
    # 两种分布(2-dist / 3-dist) 一阶导 λ* 阈值线 (斜率衰减) — 密集点横线
    if xc1_2 is not None:
        ax.axvline(x=xc1_2, color='#2E86C1', linestyle=(0, (5, 1, 1, 1)), linewidth=2.0,
                   label=f'2-dist 一阶导 λ* 阈值 x≈{xc1_2:.2f}')
    if xc1_3 is not None:
        ax.axvline(x=xc1_3, color='#FFA500', linestyle=(0, (5, 1, 1, 1)), linewidth=2.0,
                   label=f'3-dist 一阶导 λ* 阈值 x≈{xc1_3:.2f}')
    
    ax.set_xlabel('Error Bit Count', fontsize=12)
    ax.set_ylabel('Page Count', fontsize=12)
    ax.set_title('Error Bit Count Distribution', fontsize=14, pad=20)
    ax.set_xlim(-5, right_bound + 5)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Fig2: {output_path}")

# NOTE: distribution_fit.png (Fig3) 已按要求移除 —— 单分布拟合对比图不再生成。

def process_file(filename, config):
    print(f"\n{'='*60}")
    print(f"Processing file: {filename}")
    print(f"{'='*60}")
    
    group_size = determine_group_size(filename)
    max_value = config['max_value']
    print(f"  Read mode: {group_size} byte(s)")
    print(f"  Data threshold: <= {max_value}")
    
    data, over_data = read_bin_file(filename, group_size, max_value)
    if data is None:
        print(f"  x Failed to read file")
        return
    
    if not data:
        print(f"  x No valid data in file")
        return
    
    print(f"  Valid data points (<= {max_value}): {len(data):,}")
    print(f"  Over-threshold data points (> {max_value}): {len(over_data):,}")
    
    max_val = max(data)
    min_val = min(data)
    
    print(f"  Valid data range: [{min_val}, {max_val}]")
    
    if over_data:
        over_mean, over_std = welford_online_mean_std(over_data)
        print(f"  Over-limit stats: count={len(over_data):,}, mean={over_mean:.2f}, std={over_std:.2f}")
    
    stats = count_statistics(data)
    if stats is None:
        print(f"  x Statistics calculation failed")
        return
    
    thresholds = calculate_thresholds(stats, config)
    percentages = calculate_percentages(stats['counts'], thresholds)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = f"result_{timestamp}_{os.path.splitext(filename)[0]}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Output directory: {output_dir}")
    
    # 先拟合两种分布，并各自计算二阶导 λ* 阈值（数据坐标）
    mixture_result_3dist = fit_mixture_em_3dist(stats['counts'])
    mixture_result_2dist = fit_mixture_em_2dist(stats['counts'])
    right_bound = get_auto_bounds(stats['counts'])[1]
    xc3 = _compute_deriv_threshold(mixture_result_3dist, 3, right_bound)
    xc2 = _compute_deriv_threshold(mixture_result_2dist, 2, right_bound)
    xc1_3 = _compute_deriv1_threshold(mixture_result_3dist, 3, right_bound)
    xc1_2 = _compute_deriv1_threshold(mixture_result_2dist, 2, right_bound)
    
    # Fig4: 3-dist EM mixture (Gamma+Gamma+Lognormal) —— 重命名 mixture_fit.png -> mixture_fit3.png
    plot_mixture_fit_3dist(stats['counts'], mixture_result_3dist, thresholds,
                           os.path.join(output_dir, 'mixture_fit3.png'))
    
    # Fig5: 2-dist EM mixture (Lognormal+Gamma) — Control
    plot_mixture_fit_2dist(stats['counts'], mixture_result_2dist, thresholds, os.path.join(output_dir, 'mixture_fit2.png'))
    
    # Fig6/Fig7: mixture_newparams_annotated —— 结果上移一级目录，与 count_distribution.png 等并列
    # 数值来源为 BIN 文件前置统计 (均值, 标准差)
    plot_mixture_newparams_annotated(stats['counts'], mixture_result_2dist, 2, stats,
                                     os.path.join(output_dir, 'mixture_newparams_2dist.png'))
    
    plot_mixture_newparams_annotated(stats['counts'], mixture_result_3dist, 3, stats,
                                     os.path.join(output_dir, 'mixture_newparams_3dist.png'))
    
    # Fig2: count_distribution 最后绘制，纳入两种分布(2-dist/3-dist) 的 λ* 阈值线
    plot_count_distribution(stats['counts'], thresholds, percentages,
                          os.path.join(output_dir, 'count_distribution.png'),
                          xc2=xc2, xc3=xc3, xc1_2=xc1_2, xc1_3=xc1_3, config=config)
    
    if over_data:
        over_mean, over_std = welford_online_mean_std(over_data)
        over_min = min(over_data)
        over_max = max(over_data)
    else:
        over_mean = over_std = over_min = over_max = 0
    
    scan_info_path = os.path.join(output_dir, 'scanInfo.txt')
    with open(scan_info_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"Scan Information Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"\nFile Name: {filename}\n")
        f.write(f"Read Mode: {group_size} byte(s)\n")
        f.write(f"Data Threshold: <= {max_value}\n")
        f.write(f"\n[Valid Data Statistics]\n")
        f.write(f"  Total Pages: {len(data):,}\n")
        f.write(f"  Min Value: {min_val}\n")
        f.write(f"  Max Value: {max_val}\n")
        f.write(f"  Mean: {stats['mean']:.4f}\n")
        f.write(f"  Std Dev: {stats['std']:.4f}\n")
        f.write(f"\n[Threshold Lines]\n")
        f.write(f"  Mean: {thresholds['mean']:.4f} ({percentages['mean_pct']:.2f}% <=)\n")
        f.write(f"  Mean + {config['std_multiplier']}xStd: {thresholds['mean_plus_std']:.4f} ({percentages['mean_plus_std_pct']:.2f}% <=)\n")
        f.write(f"\n[Derivative λ* Thresholds (二阶导 λ*=10%, 一阶导 λ*=5%)]\n")
        if xc3 is not None:
            f.write(f"  3-dist 二阶导 λ* (曲率衰减): {xc3:.4f}\n")
        if xc2 is not None:
            f.write(f"  2-dist 二阶导 λ* (曲率衰减): {xc2:.4f}\n")
        if xc1_3 is not None:
            f.write(f"  3-dist 一阶导 λ* (斜率衰减): {xc1_3:.4f}\n")
        if xc1_2 is not None:
            f.write(f"  2-dist 一阶导 λ* (斜率衰减): {xc1_2:.4f}\n")
        f.write(f"\n[Over-limit Data Statistics (> {max_value})]\n")
        f.write(f"  Total Pages: {len(over_data):,}\n")
        f.write(f"  Min Value: {over_min}\n")
        f.write(f"  Max Value: {over_max}\n")
        f.write(f"  Mean: {over_mean:.4f}\n")
        f.write(f"  Std Dev: {over_std:.4f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"  Scan info saved to: {scan_info_path}")
    print(f"  v Processing complete")

def main():
    global config
    config = load_config()
    
    print(f"\nECC Data Analysis Tool")
    print(f"Config: MAD_mult={config['mad_multiplier']}, Std_mult={config['std_multiplier']}, MAD_scale={config['mad_scale']}, Data_threshold={config['max_value']}")
    
    bin_files = find_bin_files()
    
    if not bin_files:
        print("\nError: No .bin files found in current directory")
        print("Please place BIN files in the current directory and run again")
        _pause()
        sys.exit(1)
    
    print(f"\nFound {len(bin_files)} BIN file(s):")
    for i, f in enumerate(bin_files, 1):
        print(f"  {i}. {f}")
    
    for filename in bin_files:
        try:
            process_file(filename, config)
        except Exception as e:
            print(f"\nError processing file {filename}: {e}")
            import traceback
            traceback.print_exc()
            print("\nProgram will terminate")
            _pause()
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("v All files processed successfully")
    print(f"{'='*60}")
    _pause()

if __name__ == "__main__":
    main()