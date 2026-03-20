#!/usr/bin/env python3
"""
解压市场数据 zip 文件。

数据提供商使用 AES-256 加密，密码由文件名 + 盐值的 SHA256 生成。

用法:
    # 解压单个文件
    python scripts/unzip_market_data.py /path/to/20260309.zip /path/to/output/

    # 解压整个目录下所有 zip
    python scripts/unzip_market_data.py /path/to/zip_dir/ /path/to/output/

    # 解压 A股 + 美股 202603 月份数据
    python scripts/unzip_market_data.py --month 202603
"""

import hashlib
import os
import sys
import glob
import argparse

# 固定的盐值
SALT = "vvtr123!@#qwe"

# 数据目录常量
CN_BASE = "/Users/bertwang/Cursor/A股"
US_BASE = "/Users/bertwang/Cursor/美股数据"


def generate_zip_password(filename):
    """根据文件名和盐值生成 SHA256 加密密码"""
    data_to_hash = f"{filename}{SALT}".encode('utf-8')
    return hashlib.sha256(data_to_hash).hexdigest()


def unzip_file(zip_path, output_dir):
    """解压单个 zip 文件到目标目录"""
    import pyzipper

    filename = os.path.basename(zip_path)
    password = generate_zip_password(filename)

    os.makedirs(output_dir, exist_ok=True)

    try:
        with pyzipper.AESZipFile(zip_path, 'r') as zf:
            zf.setpassword(password.encode('utf-8'))
            zf.extractall(output_dir)
        print(f"  ✓ {filename}")
        return True
    except Exception as e:
        print(f"  ✗ {filename}: {e}")
        return False


def unzip_directory(zip_dir, output_dir):
    """解压目录下所有 zip 文件"""
    zip_files = sorted(glob.glob(os.path.join(zip_dir, "*.zip")))
    # 去重：如果有 20260302.zip 和 20260302(1).zip，只取不带括号的
    seen_dates = set()
    unique_zips = []
    for zf in zip_files:
        base = os.path.basename(zf)
        # Extract date part: 20260302 from 20260302.zip or 20260302(1).zip
        date_part = base.split('(')[0].replace('.zip', '')
        if date_part not in seen_dates:
            seen_dates.add(date_part)
            # Prefer the one without parentheses
            clean_name = f"{date_part}.zip"
            clean_path = os.path.join(zip_dir, clean_name)
            if os.path.exists(clean_path):
                unique_zips.append(clean_path)
            else:
                unique_zips.append(zf)

    # Check which ones are already extracted
    to_extract = []
    for zf in unique_zips:
        base = os.path.basename(zf).replace('.zip', '')
        csv_path = os.path.join(output_dir, f"{base}.csv")
        if not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0:
            to_extract.append(zf)

    if not to_extract:
        print(f"  所有文件已解压 ({len(unique_zips)} files)")
        return len(unique_zips), 0

    print(f"  需要解压 {len(to_extract)}/{len(unique_zips)} 个文件")
    success = 0
    for zf in to_extract:
        if unzip_file(zf, output_dir):
            success += 1

    return success, len(to_extract) - success


def unzip_month(month, markets=None):
    """解压指定月份的 A股 和 美股 数据"""
    if markets is None:
        markets = ['CN', 'US']

    results = {}
    for market in markets:
        if market == 'CN':
            zip_dir = os.path.join(CN_BASE, "1d", month)
            out_dir = os.path.join(CN_BASE, "1d_unzip", month)
            label = "A股"
        else:
            zip_dir = os.path.join(US_BASE, "1d", month)
            out_dir = os.path.join(US_BASE, "1d_unzip", month)
            label = "美股"

        if not os.path.exists(zip_dir):
            print(f"⚠️  {label} {month} 目录不存在: {zip_dir}")
            continue

        print(f"\n📦 解压 {label} {month}:")
        ok, fail = unzip_directory(zip_dir, out_dir)
        results[market] = (ok, fail)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="解压市场数据")
    parser.add_argument("source", nargs="?", help="zip 文件或目录路径")
    parser.add_argument("output", nargs="?", help="输出目录")
    parser.add_argument("--month", help="按月份解压，如 202603")
    parser.add_argument("--market", choices=['CN', 'US'], help="指定市场")

    args = parser.parse_args()

    if args.month:
        markets = [args.market] if args.market else None
        unzip_month(args.month, markets)
    elif args.source:
        if not args.output:
            print("错误: 需要指定输出目录")
            sys.exit(1)
        if os.path.isdir(args.source):
            unzip_directory(args.source, args.output)
        else:
            unzip_file(args.source, args.output)
    else:
        parser.print_help()
