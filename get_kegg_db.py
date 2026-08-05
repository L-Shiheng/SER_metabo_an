#!/usr/bin/env python3
"""
KEGG 通路-化合物数据抓取脚本（符合 KEGG API 使用规范）
用法: python get_kegg_db.py [物种代码]   # 默认 hsa
输出: kegg_<物种代码>.csv
"""

import requests
import pandas as pd
import sys
import re
import time
import json
import os
from urllib.parse import urlparse

# ---------- 配置 ----------
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
REQUEST_DELAY = 1.0          # 每次请求后等待秒数（远低于 3 req/s 限制）
MAX_RETRIES = 5              # 最大重试次数
CACHE_FILE = "kegg_cpd_names.json"  # 化合物字典缓存文件

# ---------- 工具函数 ----------
def safe_request(url, retries=MAX_RETRIES, timeout=30):
    """
    带重试机制的 GET 请求，遇到错误时指数退避
    """
    for attempt in range(1, retries + 1):
        try:
            headers = {'User-Agent': USER_AGENT}
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()  # 非 2xx 状态码抛出异常
            return resp.text
        except requests.exceptions.RequestException as e:
            print(f"⚠️  请求失败 (尝试 {attempt}/{retries}): {e}")
            if attempt == retries:
                raise  # 重试耗尽，向上抛出
            # 指数退避：2^attempt 秒，但最多 30 秒
            sleep_time = min(2 ** attempt, 30)
            time.sleep(sleep_time)
    raise RuntimeError(f"无法获取 {url}")

def download_cpd_names(force_refresh=False):
    """
    下载或从缓存加载化合物字典
    返回 dict: {cpd_id: compound_name}
    """
    if not force_refresh and os.path.exists(CACHE_FILE):
        print(f"📂 发现缓存文件 {CACHE_FILE}，正在加载...")
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)

    print("📥 正在下载全库化合物字典（约 2 万条，可能需要数十秒）...")
    text = safe_request("https://rest.kegg.jp/list/cpd")
    cpd_names = {}
    for line in text.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            cpd_id = parts[0].replace('cpd:', '')
            # 取第一个分号前的名称（通常是通用名）
            name = parts[1].split(';')[0].strip()
            cpd_names[cpd_id] = name

    # 保存到缓存
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cpd_names, f, ensure_ascii=False, indent=2)
    print(f"✅ 化合物字典已缓存到 {CACHE_FILE}")
    return cpd_names

def fetch_kegg_database(species_code):
    print(f"⏳ 开始获取物种 {species_code} 的 KEGG 通路-化合物数据...")
    # 为避免触发限流，统一在每次请求后休眠
    def req_and_delay(url):
        text = safe_request(url)
        time.sleep(REQUEST_DELAY)
        return text

    # 1. 获取该物种的通路列表
    print("📥 1/3 获取通路列表...")
    text1 = req_and_delay(f"https://rest.kegg.jp/list/pathway/{species_code}")
    pw_names = {}
    for line in text1.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) == 2:
            pw_id = parts[0].replace('path:', '')   # 如 'hsa00010'
            # 保存完整ID，同时提取数字部分用于后续映射键
            pw_names[pw_id] = parts[1]

    # 2. 获取化合物字典（优先缓存）
    print("📥 2/3 获取化合物字典...")
    cpd_names = download_cpd_names()
    # 字典下载已经包含了请求，但我们也应该在此处延迟（已自带，但再确保）
    time.sleep(REQUEST_DELAY)

    # 3. 获取全库通路-化合物映射，并按物种过滤
    print("📥 3/3 获取全库通路-化合物映射，并筛选目标物种...")
    text3 = req_and_delay("https://rest.kegg.jp/link/cpd/pathway")
    pw_cpd_map = {}  # key: 数字ID (如 '00010'), value: 化合物名称列表

    for line in text3.strip().split('\n'):
        if not line:
            continue
        parts = line.split('\t')
        if len(parts) != 2:
            continue
        # 格式: cpd:C00001 \t path:hsa00010
        cpd_part = parts[0]
        path_part = parts[1]
        if not path_part.startswith('path:'):
            continue
        pw_full = path_part.replace('path:', '')   # 'hsa00010'
        # 只处理属于目标物种的通路
        if not pw_full.startswith(species_code):
            continue
        # 提取数字部分，如 '00010'
        pw_num = re.sub(r'^[a-z]+', '', pw_full)
        cpd_id = cpd_part.replace('cpd:', '')
        cpd_name = cpd_names.get(cpd_id, cpd_id)   # 若缺失则用ID本身
        pw_cpd_map.setdefault(pw_num, []).append(cpd_name)

    # 4. 组装结果（只保留在通路列表中有映射的通路）
    print("💾 正在组装数据并保存...")
    links = []
    for pw_id, name in pw_names.items():
        pw_num = re.sub(r'^[a-z]+', '', pw_id)
        if pw_num in pw_cpd_map:
            # 去重（有些化合物可能在同一条通路重复出现，但一般不会）
            compounds = list(dict.fromkeys(pw_cpd_map[pw_num]))  # 保持顺序去重
            links.append({
                "Pathway": name,
                "Compounds": ';'.join(compounds)
            })

    # 排序（可选：按通路ID排序）
    # links.sort(key=lambda x: x["Pathway"])

    df = pd.DataFrame(links)
    filename = f"kegg_{species_code}.csv"
    df.to_csv(filename, index=False, encoding='utf-8-sig')  # 支持Excel中文
    print(f"✅ 大功告成！共抓取 {len(df)} 条含化合物的通路，文件已保存为: {filename}")

if __name__ == "__main__":
    species = sys.argv[1] if len(sys.argv) > 1 else "hsa"
    # 检查是否是合法物种代码（简单检查，只允许字母数字）
    if not re.match(r'^[a-zA-Z0-9]+$', species):
        print(f"❌ 物种代码 '{species}' 不合法，仅允许字母数字")
        sys.exit(1)
    fetch_kegg_database(species)
