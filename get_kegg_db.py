import urllib.request
import pandas as pd
import re
import sys
import time
import xml.etree.ElementTree as ET

SPECIES_MAP = {
    "hsa": "Human (Homo sapiens)",
    "mmu": "Mouse (Mus musculus)",
    "rno": "Rat (Rattus norvegicus)",
    "map": "General (reference pathway)",
}
METABOLISM_MAX_NUM = 2000
MAX_RETRY = 3

def parse_kgml_compounds(species_code, pw_num, headers):
    """从 KGML 文件中解析化合物 KEGG ID（纯 C number）"""
    url = f"https://rest.kegg.jp/get/{species_code}{pw_num}/kgml"
    compounds = set()
    for attempt in range(1, MAX_RETRY + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            for entry in root.findall('entry'):
                if entry.get('type') == 'compound':
                    name_attr = entry.get('name', '')
                    for name in name_attr.split():
                        if name.startswith('cpd:'):
                            compounds.add(name.replace('cpd:', ''))
            return compounds
        except Exception:
            if attempt < MAX_RETRY:
                time.sleep(2)
            else:
                return compounds
    return compounds

def fetch_kegg_database(species_code="map"):
    if species_code == "all":
        for code in SPECIES_MAP:
            fetch_kegg_database(code)
        return

    if species_code not in SPECIES_MAP:
        print(f"❌ 未知物种代码: {species_code}，可选: {', '.join(SPECIES_MAP)}")
        return

    species_name = SPECIES_MAP[species_code]
    print(f"⏳ 正在连接 KEGG 官方服务器（{species_name} [{species_code}]）...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    out_filename = f"kegg_{species_code}.csv"

    print("📥 1/2 正在下载代谢通路列表...")
    pw_names_all = {}
    req1 = urllib.request.Request(f"https://rest.kegg.jp/list/pathway/{species_code}", headers=headers)
    with urllib.request.urlopen(req1, timeout=30) as response:
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                pw_id = parts[0].replace('path:', '')
                pw_num = re.sub(r'^[a-z]+', '', pw_id)
                pw_names_all[pw_num] = parts[1]

    pw_names = {}
    for pw_num, pw_name in pw_names_all.items():
        try:
            if int(pw_num) < METABOLISM_MAX_NUM:
                pw_names[pw_num] = pw_name
        except ValueError:
            continue
    print(f"   → 筛选后保留 {len(pw_names)} 条纯代谢通路")

    pw_nums = list(pw_names.keys())
    total = len(pw_nums)
    print(f"📥 2/2 正在解析 KGML 树获取纯 KEGG ID 集合（共 {total} 条通路）...")

    records = []
    fail_count = 0
    for idx, pw_num in enumerate(pw_nums):
        pw_name = pw_names[pw_num]
        compounds = parse_kgml_compounds(species_code, pw_num, headers)
        if compounds:
            # 💡 核心修正：仅保留纯粹的 C number，确保在 app.py 交叉比对时 100% 精确咬合
            records.append({
                "Pathway": pw_name,
                "Compounds": ";".join(sorted(compounds))
            })
        else:
            fail_count += 1

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"   → 进度: {idx+1}/{total}（成功 {len(records)}，失败 {fail_count}）")

    if not records:
        print("❌ 错误：未能解析到化合物，请重试。")
        return

    print("💾 正在整理并保存...")
    df = pd.DataFrame(records)
    df.to_csv(out_filename, index=False)
    print(f"✅ {species_name}: 已抓取 {len(df)} 条代谢通路。")

if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "map"
    fetch_kegg_database(code)
