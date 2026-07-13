import os
import struct
import sys
import configparser
from datetime import datetime
import numpy as np
from scipy import stats
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

def filter_by_groups(data, threshold):
    filtered_data = []
    group_size = 4
    
    for i in range(0, len(data), group_size):
        group = data[i:i+group_size]
        if len(group) < group_size:
            continue
        
        has_outlier = False
        for value in group:
            if value > threshold:
                has_outlier = True
                break
        
        if not has_outlier:
            filtered_data.extend(group)
    
    return filtered_data

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

def plot_count_distribution(original_counts, filtered_counts, thresholds, percentages, output_path):
    left_bound, right_bound = get_auto_bounds(original_counts)
    max_bit = right_bound
    x = list(range(max_bit + 1))
    
    original_counts_padded = np.pad(original_counts[:max_bit + 1], (0, max_bit + 1 - len(original_counts[:max_bit + 1])), mode='constant')
    filtered_counts_padded = np.pad(filtered_counts[:max_bit + 1], (0, max_bit + 1 - len(filtered_counts[:max_bit + 1])), mode='constant')
    
    fig, (summary_ax, ax) = plt.subplots(2, 1, figsize=(14, 11), dpi=100,
                                          gridspec_kw={'height_ratios': [1.5, 6]})
    
    summary_ax.axis('off')
    
    stats_text = f"""
Statistics Summary:
─────────────────────────────────────────────────────────────
Original Page Count: {sum(original_counts_padded):,}
Filtered Page Count: {sum(filtered_counts_padded):,}

Median Series:
  Median: {thresholds['median']:.4f} ({percentages['median_pct']:.2f}% ≤)
  Median + {config['mad_multiplier']}x{config['mad_scale']}xMAD: {thresholds['median_plus_mad']:.4f} ({percentages['median_plus_mad_pct']:.2f}% ≤)

Mean Series:
  Mean: {thresholds['mean']:.4f} ({percentages['mean_pct']:.2f}% ≤)
  Mean + {config['std_multiplier']}xStd: {thresholds['mean_plus_std']:.4f} ({percentages['mean_plus_std_pct']:.2f}% ≤)

Filter Threshold: Mean + {config['std_multiplier']}xStd = {thresholds['mean_plus_std']:.4f}
Groups filtered out: {(sum(original_counts_padded) - sum(filtered_counts_padded)) // 4:,}
"""
    
    summary_ax.text(0.02, 0.95, stats_text, ha='left', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    ax.bar(x, original_counts_padded, color='#1f77b4', edgecolor='#1565c0', alpha=0.5, width=1.0, label='Original')
    ax.bar(x, filtered_counts_padded, color='#ff7f0e', edgecolor='#d35400', alpha=0.5, width=1.0, label='Filtered')
    
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
    ax.set_title('Error Bit Count Distribution (Original vs Filtered)', fontsize=14, pad=20)
    ax.set_xlim(-5, right_bound + 5)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  Fig1: {output_path}")

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
    
    filter_threshold = thresholds['mean_plus_std']
    print(f"  Filter threshold: Mean + {config['std_multiplier']}xStd = {filter_threshold:.4f}")
    
    filtered_data = filter_by_groups(data, filter_threshold)
    print(f"  Filtered data points: {len(filtered_data):,}")
    print(f"  Groups filtered out: {(len(data) - len(filtered_data)) // 4:,}")
    
    filtered_stats = count_statistics(filtered_data)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = f"result_{timestamp}_{os.path.splitext(filename)[0]}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  Output directory: {output_dir}")
    
    plot_count_distribution(stats['counts'], filtered_stats['counts'], thresholds, percentages, 
                            os.path.join(output_dir, 'count_distribution.png'))
    
    if over_data:
        over_mean, over_std = welford_online_mean_std(over_data)
        over_min = min(over_data)
        over_max = max(over_data)
    else:
        over_mean = over_std = over_min = over_max = 0
    
    scan_info_path = os.path.join(output_dir, 'scanInfo.txt')
    with open(scan_info_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"QLC Data Analysis Report\n")
        f.write(f"{'='*60}\n")
        f.write(f"\nFile Name: {filename}\n")
        f.write(f"Read Mode: {group_size} byte(s)\n")
        f.write(f"Data Threshold: <= {max_value}\n")
        f.write(f"\n[Original Data Statistics]\n")
        f.write(f"  Total Pages: {len(data):,}\n")
        f.write(f"  Min Value: {min_val}\n")
        f.write(f"  Max Value: {max_val}\n")
        f.write(f"  Mean: {stats['mean']:.4f}\n")
        f.write(f"  Std Dev: {stats['std']:.4f}\n")
        f.write(f"  Median: {stats['median']}\n")
        f.write(f"  MAD: {stats['mad']:.4f}\n")
        f.write(f"\n[Filtered Data Statistics]\n")
        f.write(f"  Filter Method: Group filtering (4 data points per group)\n")
        f.write(f"  Filter Threshold: Mean + {config['std_multiplier']}xStd = {filter_threshold:.4f}\n")
        f.write(f"  Groups filtered out: {(len(data) - len(filtered_data)) // 4:,}\n")
        f.write(f"  Total Pages: {len(filtered_data):,}\n")
        f.write(f"  Mean: {filtered_stats['mean']:.4f}\n")
        f.write(f"  Std Dev: {filtered_stats['std']:.4f}\n")
        f.write(f"  Median: {filtered_stats['median']}\n")
        f.write(f"  MAD: {filtered_stats['mad']:.4f}\n")
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
    
    print(f"\nQLC Data Analysis Tool")
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