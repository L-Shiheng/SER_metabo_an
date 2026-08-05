import urllib.request
import time
import pandas as pd

def fetch_kegg_database():
    print("⏳ 正在连接 KEGG 官方服务器...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    
    # 1. 获取所有代谢通路 (Map)
    print("📥 1/3 正在下载通路列表...")
    req1 = urllib.request.Request("http://rest.kegg.jp/list/pathway/map", headers=headers)
    with urllib.request.urlopen(req1) as response:
        pw_names = {}
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                pw_id = parts[0].replace('path:', '')
                pw_names[pw_id] = parts[1]
    time.sleep(0.5)  # 避免请求过快

    # 2. 获取所有化合物 (Compound)
    print("📥 2/3 正在下载代谢物字典...")
    req2 = urllib.request.Request("http://rest.kegg.jp/list/compound", headers=headers)
    with urllib.request.urlopen(req2) as response:
        cpd_names = {}
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                cpd_id = parts[0].replace('cpd:', '')
                first_name = parts[1].split(';')[0].strip()
                cpd_names[cpd_id] = first_name
    time.sleep(0.5)

    # 3. 获取 通路-化合物 映射关系 (关键修复点)
    print("📥 3/3 正在下载通路-代谢物映射关系...")
    # 修复：将命令从 "cpd/pathway" 改为 "pathway/cpd"
    req3 = urllib.request.Request("http://rest.kegg.jp/link/pathway/cpd", headers=headers) 
    with urllib.request.urlopen(req3) as response:
        links = []
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                # 注意：顺序也变了，第一个是通路，第二个是化合物
                pw_id = parts[0].replace('path:', '') 
                cpd_id = parts[1].replace('cpd:', '')
                
                if pw_id in pw_names and cpd_id in cpd_names:
                    links.append({
                        "Pathway": pw_names[pw_id], 
                        "Metabolite": cpd_names[cpd_id]
                    })

    if not links:
        print("❌ 错误：未能拼装成功，请检查网络和命令。")
        return

    print("💾 正在整理并保存...")
    df = pd.DataFrame(links)
    df = df.dropna()
    df.to_csv("kegg_pathways.csv", index=False)
    print(f"✅ 大功告成！已成功抓取 {len(df['Pathway'].unique())} 条通路，涵盖 {len(df)} 个代谢物映射节点。")

if __name__ == "__main__":
    fetch_kegg_database()
