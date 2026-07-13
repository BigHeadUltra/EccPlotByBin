﻿import os
import struct

def read_bin_raw(file_path):
    data = []
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(2)
            if not chunk:
                break
            if len(chunk) < 2:
                chunk = chunk.ljust(2, b'\x00')
            value = struct.unpack('<H', chunk)[0]
            data.append(value)
    return data

def write_bin_raw(file_path, data):
    with open(file_path, 'wb') as f:
        for value in data:
            f.write(struct.pack('<H', value))

def split_bin_by_plane():
    current_dir = os.getcwd()
    bin_files = [f for f in os.listdir(current_dir) if f.endswith('.bin') and 'plane' not in f.lower()]
    
    if not bin_files:
        print("Error: No .bin files found in current directory")
        return
    
    print(f"Found {len(bin_files)} BIN files:")
    for idx, file in enumerate(bin_files, 1):
        print(f"{idx}. {file}")
    
    for bin_file in bin_files:
        print(f"\nProcessing: {bin_file}")
        
        data = read_bin_raw(bin_file)
        total_count = len(data)
        print(f"Total data points: {total_count}")
        
        split_size = total_count // 4
        remainder = total_count % 4
        
        parts = []
        start = 0
        for i in range(4):
            end = start + split_size + (1 if i < remainder else 0)
            parts.append(data[start:end])
            start = end
        
        base_name = bin_file[:-4]
        
        for plane_idx, part_data in enumerate(parts):
            output_name = f"{base_name}_plane{plane_idx}.bin"
            write_bin_raw(output_name, part_data)
            print(f"  -> {output_name}: {len(part_data)} data points")
        
        print(f"Successfully split into 4 plane files")

if __name__ == '__main__':
    split_bin_by_plane()
    print("\nSplit operation completed!")
