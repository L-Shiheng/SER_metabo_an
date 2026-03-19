import urllib.request
import pandas as pd

def fetch_kegg_database():
    print("⏳ 正在连接 KEGG 官方服务器...")
    
    # 1. 获取所有代谢通路 (Map) 的 ID 和 名称
    print("📥 1/3 正在下载通路列表...")
    pw_names = {}
    with urllib.request.urlopen("http://rest.kegg.jp/list/pathway/map") as response:
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                # 例如: path:map00010 -> Glycolysis / Gluconeogenesis
                pw_names[parts[0]] = parts[1]

    # 2. 获取所有化合物 (Compound) 的 ID 和 通用名
    print("📥 2/3 正在下载代谢物字典 (这可能需要十几秒)...")
    cpd_names = {}
    with urllib.request.urlopen("http://rest.kegg.jp/list/cpd") as response:
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                # KEGG的名字通常是 "Pyruvate; Pyruvic acid"，我们取分号前面的第一个常用名
                first_name = parts[1].split(';')[0].strip()
                cpd_names[parts[0]] = first_name

    # 3. 获取 通路 和 化合物 的映射关系
    print("📥 3/3 正在下载通路-代谢物映射关系...")
    links = []
    with urllib.request.urlopen("http://rest.kegg.jp/link/cpd/map") as response:
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                pw_id = parts[0]
                cpd_id = parts[1]
                # 如果这个映射在我们的字典里，就把它们组合起来
                if pw_id in pw_names and cpd_id in cpd_names:
                    links.append({
                        "Pathway": pw_names[pw_id], 
                        "Metabolite": cpd_names[cpd_id]
                    })

    # 4. 保存为 CSV
    print("💾 正在整理并保存...")
    df = pd.DataFrame(links)
    
    # 清理掉名字为空的数据
    df = df.dropna()
    
    # 保存为您的应用所需的格式
    df.to_csv("kegg_pathways.csv", index=False)
    
    print(f"✅ 大功告成！已成功抓取 {len(df['Pathway'].unique())} 条通路，涵盖 {len(df)} 个代谢物映射节点。")
    print("📁 文件已保存为当前目录下的: kegg_pathways.csv")

if __name__ == "__main__":
    fetch_kegg_database()
