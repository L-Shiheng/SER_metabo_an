import urllib.request
import pandas as pd
import re
import sys

SPECIES_MAP = {
    "hsa": "Human (Homo sapiens)",
    "mmu": "Mouse (Mus musculus)",
    "rno": "Rat (Rattus norvegicus)",
    "map": "General (reference pathway)",
}
METABOLISM_MAX_NUM = 2000

def fetch_api(url):
    """通用的 HTTPS 请求函数，带完美伪装"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'}
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as response:
        return [line for line in response.read().decode('utf-8').strip().split('\n') if line]

def fetch_kegg_database(species_code="map"):
    if species_code == "all":
        for code in SPECIES_MAP: 
            fetch_kegg_database(code)
        return

    if species_code not in SPECIES_MAP:
        print(f"❌ 未知物种代码: {species_code}")
        return

    print(f"⏳ 正在连接 KEGG 服务器，执行严谨生物学链路抓取（{SPECIES_MAP[species_code]}）...")

    # 1. 获取代谢通路列表
    pw_names = {}
    for line in fetch_api(f"https://rest.kegg.jp/list/pathway/{species_code}"):
        parts = line.split('\t')
        if len(parts) == 2:
            pw_num = re.sub(r'^[a-z]+', '', parts[0].replace('path:', ''))
            try:
                if int(pw_num) < METABOLISM_MAX_NUM:
                    pw_names[pw_num] = parts[1]
            except ValueError:
                pass
    print(f"📥 1/4 提取代谢通路总数: {len(pw_names)} 条")

    # 2. 获取该物种特异性的“合法反应”集合
    species_valid_rns = set()
    if species_code != "map":
        for line in fetch_api(f"https://rest.kegg.jp/link/rn/{species_code}"):
            parts = line.split('\t')
            if len(parts) == 2:
                species_valid_rns.add(parts[1].replace('rn:', ''))
        print(f"📥 2/4 提取物种基因特异性化学反应: {len(species_valid_rns)} 个")
    else:
        print("📥 2/4 参考库模式 (保留所有化学反应)")

    # 3. 获取所有 通路 -> 反应 的映射关系
    pw_to_rn = {}
    for line in fetch_api("https://rest.kegg.jp/link/rn/pathway"):
        parts = line.split('\t')
        if len(parts) == 2:
            pw_num = re.sub(r'^[a-z]+', '', parts[0].replace('path:', ''))
            rn = parts[1].replace('rn:', '')
            if pw_num not in pw_to_rn:
                pw_to_rn[pw_num] = []
            pw_to_rn[pw_num].append(rn)
    print(f"📥 3/4 构建 通路(Pathway) -> 反应(Reaction) 映射图")

    # 4. 获取所有 反应 -> 化合物 的映射关系
    rn_to_cpd = {}
    for line in fetch_api("https://rest.kegg.jp/link/cpd/rn"):
        parts = line.split('\t')
        if len(parts) == 2:
            rn = parts[0].replace('rn:', '')
            cpd = parts[1].replace('cpd:', '')
            if rn not in rn_to_cpd:
                rn_to_cpd[rn] = []
            rn_to_cpd[rn].append(cpd)
    print(f"📥 4/4 构建 反应(Reaction) -> 化合物(Compound) 映射图")

    # 5. 核心逻辑：生物学严谨组装
    print("💾 正在执行生物学约束组装并保存...")
    records = []
    for pw_num, name in pw_names.items():
        valid_cpds = set()
        for rn in pw_to_rn.get(pw_num, []):
            # 💡 黄金防线：仅当该反应存在于当前物种基因组中时，才提取该反应下的化合物
            if species_code == "map" or rn in species_valid_rns:
                valid_cpds.update(rn_to_cpd.get(rn, []))
        
        if valid_cpds:
            records.append({
                "Pathway": name,
                "Compounds": ";".join(sorted(valid_cpds))
            })

    if not records:
        print("❌ 错误：未能映射出任何物种特异性化合物。")
        return

    df = pd.DataFrame(records)
    out_filename = f"kegg_{species_code}.csv"
    df.to_csv(out_filename, index=False)
    
    total_cpds = sum(len(r["Compounds"].split(";")) for r in records)
    print(f"✅ {SPECIES_MAP[species_code]}: 已抓取 {len(df)} 条通路，严格过滤保留了 {total_cpds} 个物种特异性映射节点。")
    print(f"📁 文件已保存: {out_filename}")

if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "map"
    fetch_kegg_database(code)
