import os
import struct
import sys
import configparser
from datetime import datetime
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
        
        # 确保所有必要的配置项都存在
        if 'SETTINGS' not in config:
            config['SETTINGS'] = {}
        
        # 检查并补充缺失的配置项
        settings = config['SETTINGS']
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in settings:
                settings[key] = default_value
                config_updated = True
        
        # 如果有更新，保存配置文件
        if config_updated:
            with open(CONFIG_FILE, 'w') as f:
                config.write(f)
            print(f"配置文件已更新，补充了缺失的配置项")
    else:
        config['SETTINGS'] = DEFAULT_CONFIG
        with open(CONFIG_FILE, 'w') as f:
            config.write(f)
        print(f"配置文件不存在，已创建默认配置文件: {CONFIG_FILE}")
    
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
        print(f"读取文件 {file_path} 时出错: {e}")
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
  Median + {config['mad_multiplier']}×{config['mad_scale']}×MAD: {thresholds['median_plus_mad']:.4f} ({percentages['median_plus_mad_pct']:.2f}% ≤)

Mean Series:
  Mean: {thresholds['mean']:.4f} ({percentages['mean_pct']:.2f}% ≤)
  Mean + {config['std_multiplier']}×Std: {thresholds['mean_plus_std']:.4f} ({percentages['mean_plus_std_pct']:.2f}% ≤)
"""
    
    summary_ax.text(0.02, 0.95, stats_text, ha='left', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    # 使用分层抽样来提高绘图效率，确保尾部数据有足够代表性
    sample_size = 50000
    num_strata = 10  # 将数据分成10层，确保尾部数据有足够代表性
    
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
                # 如果该层数据少于需要抽取的数量，取全部
                sampled_data.extend(data_sorted[start:end])
            else:
                # 随机抽取该层的样本
                stratum_indices = random.sample(range(start, end), samples_per_stratum)
                stratum_indices.sort()
                sampled_data.extend([data_sorted[j] for j in stratum_indices])
        
        ax.scatter(range(len(sampled_data)), sampled_data, s=2, alpha=0.7, color='#1f77b4', 
                   label=f'Data (分层抽样 {len(sampled_data):,}/{len(data):,}, {num_strata}层)')
    else:
        data_sorted = sorted(data)
        ax.scatter(range(len(data_sorted)), data_sorted, s=2, alpha=0.7, color='#1f77b4', label='Data')
    
    ax.axhline(y=thresholds['median'], color='#ff7f0e', linestyle='-', linewidth=2,
               label=f'Median ({thresholds["median"]:.2f})')
    ax.axhline(y=thresholds['median_plus_mad'], color='#e377c2', linestyle='--', linewidth=2,
               label=f'Median + {config["mad_multiplier"]}×{config["mad_scale"]}×MAD ({thresholds["median_plus_mad"]:.2f})')
    ax.axhline(y=thresholds['mean'], color='#2ca02c', linestyle='-', linewidth=2,
               label=f'Mean ({thresholds["mean"]:.2f})')
    ax.axhline(y=thresholds['mean_plus_std'], color='#9467bd', linestyle='--', linewidth=2,
               label=f'Mean + {config["std_multiplier"]}×Std ({thresholds["mean_plus_std"]:.2f})')
    
    ax.set_xlabel('Sorted Page Index', fontsize=12)
    ax.set_ylabel('Error Bit Count', fontsize=12)
    ax.set_title('Sorted Error Bit Data with Statistical Thresholds', fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  图1: {output_path}")

def plot_count_distribution(counts, thresholds, percentages, output_path):
    max_bit = len(counts) - 1
    x = list(range(max_bit + 1))
    
    fig, (summary_ax, ax) = plt.subplots(2, 1, figsize=(14, 11), dpi=100,
                                          gridspec_kw={'height_ratios': [1.5, 6]})
    
    summary_ax.axis('off')
    
    stats_text = f"""
Statistics Summary:
─────────────────────────────────────────────────────────────
Page Count: {sum(counts):,}

Median Series:
  Median: {thresholds['median']:.4f} ({percentages['median_pct']:.2f}% ≤)
  Median + {config['mad_multiplier']}×{config['mad_scale']}×MAD: {thresholds['median_plus_mad']:.4f} ({percentages['median_plus_mad_pct']:.2f}% ≤)

Mean Series:
  Mean: {thresholds['mean']:.4f} ({percentages['mean_pct']:.2f}% ≤)
  Mean + {config['std_multiplier']}×Std: {thresholds['mean_plus_std']:.4f} ({percentages['mean_plus_std_pct']:.2f}% ≤)
"""
    
    summary_ax.text(0.02, 0.95, stats_text, ha='left', va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray'))
    
    ax.bar(x, counts, color='#1f77b4', edgecolor='#1565c0', alpha=0.7, width=1.0)
    
    ax.axvline(x=thresholds['median'], color='#ff7f0e', linestyle='-', linewidth=2,
               label=f'Median ({thresholds["median"]:.2f})')
    ax.axvline(x=thresholds['median_plus_mad'], color='#e377c2', linestyle='--', linewidth=2,
               label=f'Median + {config["mad_multiplier"]}×{config["mad_scale"]}×MAD')
    ax.axvline(x=thresholds['mean'], color='#2ca02c', linestyle='-', linewidth=2,
               label=f'Mean ({thresholds["mean"]:.2f})')
    ax.axvline(x=thresholds['mean_plus_std'], color='#9467bd', linestyle='--', linewidth=2,
               label=f'Mean + {config["std_multiplier"]}×Std')
    
    ax.set_xlabel('Error Bit Count', fontsize=12)
    ax.set_ylabel('Page Count', fontsize=12)
    ax.set_title('Error Bit Count Distribution', fontsize=14, pad=20)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"  图2: {output_path}")

def process_file(filename, config):
    print(f"\n{'='*60}")
    print(f"处理文件: {filename}")
    print(f"{'='*60}")
    
    group_size = determine_group_size(filename)
    max_value = config['max_value']
    print(f"  读取方式: {group_size}字节")
    print(f"  数据阈值: <= {max_value}")
    
    data, over_data = read_bin_file(filename, group_size, max_value)
    if data is None:
        print(f"  ✗ 读取文件失败")
        return
    
    if not data:
        print(f"  ✗ 文件中没有有效数据")
        return
    
    print(f"  有效数据点(<= {max_value}): {len(data):,}")
    print(f"  超过阈值的数据点(> {max_value}): {len(over_data):,}")
    
    max_val = max(data)
    min_val = min(data)
    
    print(f"  有效数据范围: [{min_val}, {max_val}]")
    
    if over_data:
        over_mean, over_std = welford_online_mean_std(over_data)
        print(f"  超限数据统计: 计数={len(over_data):,}, 均值={over_mean:.2f}, 标准差={over_std:.2f}")
    
    stats = count_statistics(data)
    if stats is None:
        print(f"  ✗ 统计计算失败")
        return
    
    thresholds = calculate_thresholds(stats, config)
    percentages = calculate_percentages(stats['counts'], thresholds)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f"result_{timestamp}_{os.path.splitext(filename)[0]}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"  输出目录: {output_dir}")
    
    plot_sorted_data(data, thresholds, percentages, os.path.join(output_dir, 'sorted_data.png'))
    plot_count_distribution(stats['counts'], thresholds, percentages, os.path.join(output_dir, 'count_distribution.png'))
    
    if over_data:
        over_mean, over_std = welford_online_mean_std(over_data)
        over_min = min(over_data)
        over_max = max(over_data)
    else:
        over_mean = over_std = over_min = over_max = 0
    
    scan_info_path = os.path.join(output_dir, 'scanInfo.txt')
    with open(scan_info_path, 'w', encoding='utf-8') as f:
        f.write(f"{'='*60}\n")
        f.write(f"扫描信息报告\n")
        f.write(f"{'='*60}\n")
        f.write(f"\n文件名称: {filename}\n")
        f.write(f"读取方式: {group_size}字节\n")
        f.write(f"数据阈值: <= {max_value}\n")
        f.write(f"\n[有效数据统计]\n")
        f.write(f"  总页数: {len(data):,}\n")
        f.write(f"  最小值: {min_val}\n")
        f.write(f"  最大值: {max_val}\n")
        f.write(f"  均值: {stats['mean']:.4f}\n")
        f.write(f"  标准差: {stats['std']:.4f}\n")
        f.write(f"  中位数: {stats['median']}\n")
        f.write(f"  MAD: {stats['mad']:.4f}\n")
        f.write(f"\n[统计阈值线]\n")
        f.write(f"  Median: {thresholds['median']:.4f} ({percentages['median_pct']:.2f}% <=)\n")
        f.write(f"  Median + {config['mad_multiplier']}×{config['mad_scale']}×MAD: {thresholds['median_plus_mad']:.4f} ({percentages['median_plus_mad_pct']:.2f}% <=)\n")
        f.write(f"  Mean: {thresholds['mean']:.4f} ({percentages['mean_pct']:.2f}% <=)\n")
        f.write(f"  Mean + {config['std_multiplier']}×Std: {thresholds['mean_plus_std']:.4f} ({percentages['mean_plus_std_pct']:.2f}% <=)\n")
        f.write(f"\n[超限数据统计 (> {max_value})]\n")
        f.write(f"  总页数: {len(over_data):,}\n")
        f.write(f"  最小值: {over_min}\n")
        f.write(f"  最大值: {over_max}\n")
        f.write(f"  均值: {over_mean:.4f}\n")
        f.write(f"  标准差: {over_std:.4f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"  扫描信息已保存到: {scan_info_path}")
    print(f"  ✓ 处理完成")

def main():
    global config
    config = load_config()
    
    print(f"\nECC 数据分析工具")
    print(f"配置参数: MAD倍数={config['mad_multiplier']}, Std倍数={config['std_multiplier']}, MAD系数={config['mad_scale']}, 数据阈值={config['max_value']}")
    
    bin_files = find_bin_files()
    
    if not bin_files:
        print("\n错误: 当前路径下没有找到.bin文件")
        print("请在当前路径下放置BIN文件后再运行")
        input("按 Enter 键退出...")
        sys.exit(1)
    
    print(f"\n找到 {len(bin_files)} 个BIN文件:")
    for i, f in enumerate(bin_files, 1):
        print(f"  {i}. {f}")
    
    for filename in bin_files:
        try:
            process_file(filename, config)
        except Exception as e:
            print(f"\n处理文件 {filename} 时发生错误: {e}")
            import traceback
            traceback.print_exc()
            print("\n程序将终止")
            input("按 Enter 键退出...")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✓ 所有文件处理完成")
    print(f"{'='*60}")
    input("按 Enter 键退出...")

if __name__ == "__main__":
    main()