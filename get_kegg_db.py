import urllib.request
import pandas as pd
import sys
import re

def fetch_kegg_database(species_code):
    print(f"⏳ 正在连接 KEGG 官方服务器 (物种: {species_code})...")
    # 核心：保留完美的浏览器伪装
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    
    # 1. 获取特定物种的通路列表
    print("📥 1/3 正在下载通路列表...")
    pw_names = {}
    req1 = urllib.request.Request(f"https://rest.kegg.jp/list/pathway/{species_code}", headers=headers)
    with urllib.request.urlopen(req1) as response:
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                # 提取纯数字ID，兼容不同物种前缀，例如 hsa00010 -> 00010
                pw_id = re.sub(r'^[a-z]+', '', parts[0].replace('path:', ''))
                pw_names[pw_id] = parts[1]

    # 2. 获取代谢物字典
    print("📥 2/3 正在下载代谢物字典...")
    cpd_names = {}
    req2 = urllib.request.Request("https://rest.kegg.jp/list/cpd", headers=headers)
    with urllib.request.urlopen(req2) as response:
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                cpd_id = parts[0].replace('cpd:', '')
                cpd_names[cpd_id] = parts[1].split(';')[0].strip()

    # 3. 获取通路-代谢物映射关系
    print("📥 3/3 正在下载通路-代谢物映射关系...")
    pw_cpd_map = {}
    req3 = urllib.request.Request("https://rest.kegg.jp/link/cpd/pathway", headers=headers)
    with urllib.request.urlopen(req3) as response:
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2 and parts[0].startswith('path:map'):
                pw_num = parts[0].replace('path:map', '')
                cpd_id = parts[1].replace('cpd:', '')
                
                # 翻译为具体名称
                cpd_name = cpd_names.get(cpd_id, cpd_id)
                if pw_num not in pw_cpd_map:
                    pw_cpd_map[pw_num] = []
                pw_cpd_map[pw_num].append(cpd_name)

    # 4. 组装并保存
    print("💾 正在整理并保存...")
    links = []
    for pw_num, name in pw_names.items():
        if pw_num in pw_cpd_map:
            links.append({
                "Pathway": name,
                "Compounds": ';'.join(pw_cpd_map[pw_num])
            })
            
    df = pd.DataFrame(links)
    filename = f"kegg_{species_code}.csv"
    df.to_csv(filename, index=False)
    print(f"✅ 大功告成！已成功抓取 {len(df)} 条通路，文件已保存为: {filename}")

if __name__ == "__main__":
    # 接收来自 app.py 传过来的物种参数，如果没有则默认 hsa
    species = sys.argv[1] if len(sys.argv) > 1 else "hsa"
    fetch_kegg_database(species)
