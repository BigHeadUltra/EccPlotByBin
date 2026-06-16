import os
import struct
import sys
import configparser
from datetime import datetime
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from distfit import distfit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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
    
    median_index = n // 2
    cumsum = 0
    median = 0
    for i in range(len(counts)):
        cumsum += counts[i]
        if cumsum > median_index:
            median = i
            break
    
    mad_counts = {}
    mad_total = 0
    for i in range(len(counts)):
        if counts[i] > 0:
            dev = abs(i - median)
            mad_counts[dev] = mad_counts.get(dev, 0) + counts[i]
            mad_total += counts[i]
    
    mad_index = mad_total // 2
    cumsum = 0
    mad = 0
    for dev in sorted(mad_counts.keys()):
        cumsum += mad_counts[dev]
        if cumsum > mad_index:
            mad = dev
            break
    
    return {
        'counts': counts,
        'total': n,
        'mean': mean,
        'std': std,
        'median': median,
        'mad': mad
    }

def calculate_thresholds(stats, config):
    mean_plus_std = stats['mean'] + config['std_multiplier'] * stats['std']
    median_plus_mad = stats['median'] + config['mad_multiplier'] * config['mad_scale'] * stats['mad']
    
    return {
        'mean': stats['mean'],
        'mean_plus_std': mean_plus_std,
        'median': stats['median'],
        'median_plus_mad': median_plus_mad
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
        'mean_plus_std_pct': (count_below(thresholds['mean_plus_std']) / total) * 100,
        'median_pct': (count_below(thresholds['median']) / total) * 100,
        'median_plus_mad_pct': (count_below(thresholds['median_plus_mad']) / total) * 100
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

def plot_mixture_fit_3dist(counts, mixture_result, output_path):
    """
    Plot histogram + 3-component EM fit (Gamma + Gamma + Lognormal) — Fig4
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
    
    ax.set_xlabel('Error Bit Count', fontsize=12)
    ax.set_ylabel('Page Count', fontsize=12)
    ax.set_title('EM Mixture Fit [3-dist: Gamma+Gamma+Lognormal]', fontsize=14, pad=20)
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
    
    weights = np.array([0.5, 0.5])
    
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

def plot_mixture_fit_2dist(counts, mixture_result, output_path):
    """
    Plot histogram + 2-component EM fit (Lognormal + Gamma) — Fig5 (Control)
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
    
    ax.set_xlabel('Error Bit Count', fontsize=12)
    ax.set_ylabel('Page Count', fontsize=12)
    ax.set_title('EM Mixture Fit [2-dist: Lognormal+Gamma] - Control', fontsize=14, pad=20)
    ax.set_xlim(-5, right_bound + 5)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Fig5: {output_path}")

def plot_sorted_data(data, thresholds, percentages, output_path):
    fig, (summary_ax, ax) = plt.subplots(2, 1, figsize=(14, 11), dpi=100,
                                          gridspec_kw={'height_ratios': [1.5, 6]})
    
    summary_ax.axis('off')
    
    stats_text = f"""
Statistics Summary:
─────────────────────────────────────────────────────────────
Page Count: {len(data):,}

Median Series:
  Median: {thresholds['median']:.4f} ({percentages['median_pct']:.2f}% ≤)
  Median + {config['mad_multiplier']}x{config['mad_scale']}xMAD: {thresholds['median_plus_mad']:.4f} ({percentages['median_plus_mad_pct']:.2f}% ≤)

Mean Series:
  Mean: {thresholds['mean']:.4f} ({percentages['mean_pct']:.2f}% ≤)
  Mean + {config['std_multiplier']}xStd: {thresholds['mean_plus_std']:.4f} ({percentages['mean_plus_std_pct']:.2f}% ≤)
"""
    
    summary_ax.text(0.02, 0.95, stats_text, ha='left', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    sample_size = 50000
    num_strata = 10
    
    if len(data) > sample_size:
        import random
        random.seed(42)
        
        data_sorted = sorted(data)
        stratum_size = len(data_sorted) // num_strata
        samples_per_stratum = sample_size // num_strata
        
        sampled_data = []
        for i in range(num_strata):
            start = i * stratum_size
            end = (i + 1) * stratum_size if i < num_strata - 1 else len(data_sorted)
            
            if end - start <= samples_per_stratum:
                sampled_data.extend(data_sorted[start:end])
            else:
                stratum_indices = random.sample(range(start, end), samples_per_stratum)
                stratum_indices.sort()
                sampled_data.extend([data_sorted[j] for j in stratum_indices])
        
        ax.scatter(range(len(sampled_data)), sampled_data, s=2, alpha=0.7, color='#1f77b4', 
                   label=f'Data (sampled {len(sampled_data):,}/{len(data):,}, {num_strata} strata)')
    else:
        data_sorted = sorted(data)
        ax.scatter(range(len(data_sorted)), data_sorted, s=2, alpha=0.7, color='#1f77b4', label='Data')
    
    ax.axhline(y=thresholds['median'], color='#ff7f0e', linestyle='-', linewidth=2,
               label=f'Median ({thresholds["median"]:.2f})')
    ax.axhline(y=thresholds['median_plus_mad'], color='#e377c2', linestyle='--', linewidth=2,
               label=f'Median + {config["mad_multiplier"]}x{config["mad_scale"]}xMAD ({thresholds["median_plus_mad"]:.2f})')
    ax.axhline(y=thresholds['mean'], color='#2ca02c', linestyle='-', linewidth=2,
               label=f'Mean ({thresholds["mean"]:.2f})')
    ax.axhline(y=thresholds['mean_plus_std'], color='#9467bd', linestyle='--', linewidth=2,
               label=f'Mean + {config["std_multiplier"]}xStd ({thresholds["mean_plus_std"]:.2f})')
    
    ax.set_xlabel('Sorted Page Index', fontsize=12)
    ax.set_ylabel('Error Bit Count', fontsize=12)
    ax.set_title('Sorted Error Bit Data with Statistical Thresholds', fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Fig1: {output_path}")

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

def plot_count_distribution(counts, thresholds, percentages, output_path):
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

Median Series:
  Median: {thresholds['median']:.4f} ({percentages['median_pct']:.2f}% ≤)
  Median + {config['mad_multiplier']}x{config['mad_scale']}xMAD: {thresholds['median_plus_mad']:.4f} ({percentages['median_plus_mad_pct']:.2f}% ≤)

Mean Series:
  Mean: {thresholds['mean']:.4f} ({percentages['mean_pct']:.2f}% ≤)
  Mean + {config['std_multiplier']}xStd: {thresholds['mean_plus_std']:.4f} ({percentages['mean_plus_std_pct']:.2f}% ≤)
"""
    
    summary_ax.text(0.02, 0.95, stats_text, ha='left', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    ax.bar(x, counts, color='#1f77b4', edgecolor='#1565c0', alpha=0.7, width=1.0)
    
    ax.axvline(x=thresholds['median'], color='#ff7f0e', linestyle='-', linewidth=2,
               label=f'Median ({thresholds["median"]:.2f})')
    ax.axvline(x=thresholds['median_plus_mad'], color='#e377c2', linestyle='--', linewidth=2,
               label=f'Median + {config["mad_multiplier"]}x{config["mad_scale"]}xMAD')
    ax.axvline(x=thresholds['mean'], color='#2ca02c', linestyle='-', linewidth=2,
               label=f'Mean ({thresholds["mean"]:.2f})')
    ax.axvline(x=thresholds['mean_plus_std'], color='#9467bd', linestyle='--', linewidth=2,
               label=f'Mean + {config["std_multiplier"]}xStd')
    
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

def plot_distribution_fit(counts, dist_results, loc, output_path):
    left_bound, right_bound = get_auto_bounds(counts)
    max_bit = right_bound
    x = list(range(max_bit + 1))
    counts = np.pad(counts[:max_bit + 1], (0, max_bit + 1 - len(counts[:max_bit + 1])), mode='constant')
    total = sum(counts)
    
    fig, (summary_ax, ax) = plt.subplots(2, 1, figsize=(14, 11), dpi=100,
                                          gridspec_kw={'height_ratios': [1.5, 6]})
    
    summary_ax.axis('off')
    
    dist_names = {
        'norm': 'Normal',
        'lognorm': 'Lognormal',
        'gamma': 'Gamma'
    }
    
    colors = {
        'norm': '#1f77b4',
        'lognorm': '#ff7f0e',
        'gamma': '#2ca02c'
    }
    
    if dist_results is not None:
        summary_lines = ["Distribution Fit Results:", "-"*60]
        for dist_name, result in dist_results.items():
            params = result['params']
            score = result['score']
            name = dist_names.get(dist_name, dist_name)
            
            if dist_name == 'norm':
                mean, std = params
                summary_lines.append(f"{name}: mean={mean:.4f}, std={std:.4f}, score={score:.4f}")
            elif dist_name == 'lognorm':
                s, loc_d, scale_d = params
                summary_lines.append(f"{name}: shape={s:.4f}, loc={loc_d:.4f}, scale={scale_d:.4f}, score={score:.4f}")
            elif dist_name == 'gamma':
                a, loc_d, scale_d = params
                summary_lines.append(f"{name}: shape={a:.4f}, loc={loc_d:.4f}, scale={scale_d:.4f}, score={score:.4f}")
        
        summary_lines.append(f"\nLocation Shift (loc): {loc:.4f}")
        summary_text = "\n".join(summary_lines)
    else:
        summary_text = "Distribution Fit Failed"
    
    summary_ax.text(0.02, 0.95, summary_text, ha='left', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    ax.bar(x, counts, color='#94a3b8', edgecolor='#64748b', alpha=0.7, width=1.0, label='Histogram')
    
    if dist_results is not None:
        x_plot = np.linspace(max(0.1, loc), right_bound, 500)
        
        for dist_name, result in dist_results.items():
            params = result['params']
            dist_color = colors.get(dist_name, '#333333')
            name = dist_names.get(dist_name, dist_name)
            
            try:
                if dist_name == 'norm':
                    mean, std = params
                    pdf = stats.norm.pdf(x_plot, loc=mean + loc, scale=std)
                elif dist_name == 'lognorm':
                    s, loc_d, scale_d = params
                    pdf = stats.lognorm.pdf(x_plot, s, loc=loc_d + loc, scale=scale_d)
                elif dist_name == 'gamma':
                    a, loc_d, scale_d = params
                    pdf = stats.gamma.pdf(x_plot, a, loc=loc_d + loc, scale=scale_d)
                
                dist_counts = pdf * total
                ax.plot(x_plot, dist_counts, color=dist_color, linestyle='-', linewidth=2,
                        label=f'{name}')
            except Exception as e:
                pass
    
    ax.set_xlabel('Error Bit Count', fontsize=12)
    ax.set_ylabel('Page Count', fontsize=12)
    ax.set_title('Error Bit Count Distribution with Multiple Distribution Fits', fontsize=14, pad=20)
    ax.set_xlim(-5, right_bound + 5)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Fig3: {output_path}")

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
    
    dist_results, loc = fit_distributions(stats['counts'])
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = f"result_{timestamp}_{os.path.splitext(filename)[0]}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Output directory: {output_dir}")
    
    plot_sorted_data(data, thresholds, percentages, os.path.join(output_dir, 'sorted_data.png'))
    plot_count_distribution(stats['counts'], thresholds, percentages, os.path.join(output_dir, 'count_distribution.png'))
    plot_distribution_fit(stats['counts'], dist_results, loc, os.path.join(output_dir, 'distribution_fit.png'))
    
    # Fig4: 3-dist EM mixture (Gamma+Gamma+Lognormal)
    mixture_result_3dist = fit_mixture_em_3dist(stats['counts'])
    plot_mixture_fit_3dist(stats['counts'], mixture_result_3dist, os.path.join(output_dir, 'mixture_fit.png'))
    
    # Fig5: 2-dist EM mixture (Lognormal+Gamma) — Control
    mixture_result_2dist = fit_mixture_em_2dist(stats['counts'])
    plot_mixture_fit_2dist(stats['counts'], mixture_result_2dist, os.path.join(output_dir, 'mixture_fit2.png'))
    
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
        f.write(f"  Median: {stats['median']}\n")
        f.write(f"  MAD: {stats['mad']:.4f}\n")
        f.write(f"\n[Threshold Lines]\n")
        f.write(f"  Median: {thresholds['median']:.4f} ({percentages['median_pct']:.2f}% <=)\n")
        f.write(f"  Median + {config['mad_multiplier']}x{config['mad_scale']}xMAD: {thresholds['median_plus_mad']:.4f} ({percentages['median_plus_mad_pct']:.2f}% <=)\n")
        f.write(f"  Mean: {thresholds['mean']:.4f} ({percentages['mean_pct']:.2f}% <=)\n")
        f.write(f"  Mean + {config['std_multiplier']}xStd: {thresholds['mean_plus_std']:.4f} ({percentages['mean_plus_std_pct']:.2f}% <=)\n")
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
        input("Press Enter to exit...")
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
            input("Press Enter to exit...")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("v All files processed successfully")
    print(f"{'='*60}")
    input("Press Enter to exit...")

if __name__ == "__main__":
    main()