import urllib.request
import pandas as pd
import re
import sys
import time
import xml.etree.ElementTree as ET

# 内置 5 大研究领域的代表性物种预设（同时支持传入任意合法的 KEGG 物种代码）
SPECIES_MAP = {
    # 1. 哺乳动物与医学模式生物
    "hsa": "Human (Homo sapiens - 人类)",
    "mmu": "Mouse (Mus musculus - 小鼠)",
    "rno": "Rat (Rattus norvegicus - 大鼠)",
    "ssc": "Pig (Sus scrofa - 猪)",
    "bta": "Cow (Bos taurus - 牛)",
    "cfa": "Dog (Canis familiaris - 犬)",
    # 2. 植物、农作物与中药研究模式植物
    "ath": "Arabidopsis (Arabidopsis thaliana - 拟南芥/植物通用基准)",
    "osa": "Rice (Oryza sativa - 水稻)",
    "zma": "Maize (Zea mays - 玉米)",
    "gmx": "Soybean (Glycine max - 大豆)",
    "sly": "Tomato (Solanum lycopersicum - 番茄)",
    # 3. 环境毒理与经典模式动物
    "dre": "Zebrafish (Danio rerio - 斑马鱼)",
    "dme": "Fruit fly (Drosophila melanogaster - 果蝇)",
    "cel": "Nematode (Caenorhabditis elegans - 秀丽隐杆线虫)",
    # 4. 微生物、肠道菌群与发酵工程
    "eco": "E. coli (Escherichia coli K-12 - 大肠杆菌)",
    "sce": "Yeast (Saccharomyces cerevisiae - 酿酒酵母)",
    "bsu": "B. subtilis (Bacillus subtilis - 枯草芽孢杆菌)",
    # 5. 通用参考全库
    "map": "General Reference (跨物种全集参考通路)",
}

# 核心生化代谢通路最大编号 (< 01100 为纯正代谢子通路，剔除 01100~01299 全局总览巨型图)
METABOLISM_MAX_NUM = 1100

# 代谢组学高价值“生理/转运/信号/细胞过程”通路白名单（富含关键小分子代谢物节点）
METABOLITE_SIGNALING_WHITELIST = {
    "02010",  # ABC transporters (ABC转运蛋白：跨膜转运氨基酸/糖/脂质/核苷酸)
    "02020",  # Two-component system (双组分系统：微生物与植物环境感应)
    "02060",  # Phosphotransferase system (PTS - 细菌糖转运系统)
    "04020",  # Calcium signaling pathway (钙信号通路)
    "04022",  # cGMP-PKG signaling pathway
    "04024",  # cAMP signaling pathway (cAMP信号通路)
    "04066",  # HIF-1 signaling pathway (缺氧诱导因子信号通路)
    "04068",  # FoxO signaling pathway
    "04070",  # Phosphatidylinositol signaling system (磷脂酰肌醇信号系统)
    "04072",  # Phospholipase D signaling pathway (磷脂酶D信号通路)
    "04075",  # Plant hormone signal transduction (植物激素信号转导：生长素/茉莉酸/水杨酸/脱落酸等)
    "04080",  # Neuroactive ligand-receptor interaction (神经活性配体-受体相互作用：GABA/谷氨酸/多巴胺等)
    "04150",  # mTOR signaling pathway (mTOR氨基酸/能量感应信号通路)
    "04152",  # AMPK signaling pathway (AMPK能量感应信号通路)
    "04216",  # Ferroptosis (铁死亡：GSH/GSSG、花生四烯酸、脂质过氧化)
    "04724",  # Glutamatergic synapse (谷氨酸能突触)
    "04725",  # Cholinergic synapse (胆碱能突触)
    "04726",  # Serotonergic synapse (5-羟色胺能突触)
    "04727",  # GABAergic synapse (GABA能突触)
    "04728",  # Dopaminergic synapse (多巴胺能突触)
    "04910",  # Insulin signaling pathway (胰岛素信号通路)
    "04922",  # Glucagon signaling pathway (胰高血糖素信号通路)
    "04931",  # Insulin resistance (胰岛素抵抗)
    "04973",  # Carbohydrate digestion and absorption (碳水化合物消化吸收)
    "04974",  # Protein digestion and absorption (蛋白质与氨基酸消化吸收)
    "04975",  # Fat digestion and absorption (脂肪消化吸收)
    "04976",  # Bile secretion (胆汁分泌与胆汁酸循环)
    "04977",  # Vitamin digestion and absorption (维生素消化吸收)
    "04978",  # Mineral absorption (矿物质吸收)
    "04979",  # Cholesterol metabolism (胆固醇代谢)
    "04980",  # Cobalamin transport and metabolism (钴胺素/维生素B12转运与代谢)
}

MAX_RETRY = 3

def extract_compounds_from_root(root, prefer_reactions=True):
    """从 KGML XML 树中提取 KEGG 化合物 C-number"""
    compounds = set()
    if prefer_reactions:
        for reaction in root.findall('reaction'):
            for sub in reaction.findall('substrate'):
                for name in sub.get('name', '').split():
                    if name.startswith('cpd:'):
                        compounds.add(name.replace('cpd:', ''))
            for prod in reaction.findall('product'):
                for name in prod.get('name', '').split():
                    if name.startswith('cpd:'):
                        compounds.add(name.replace('cpd:', ''))

    # 若为生理/信号/转运通路，或该代谢通路无 <reaction> 标签，则提取 <entry type="compound">
    if not compounds:
        for entry in root.findall('entry'):
            if entry.get('type') == 'compound':
                for name in entry.get('name', '').split():
                    if name.startswith('cpd:'):
                        compounds.add(name.replace('cpd:', ''))
    return compounds

def parse_kgml_pathway(species_code, pw_num, is_core_metabolism, headers):
    """请求并解析单条 KEGG 通路的 KGML 文件"""
    target_prefix = ("rn" if is_core_metabolism else "map") if species_code == "map" else species_code
    url = f"https://rest.kegg.jp/get/{target_prefix}{pw_num}/kgml"
    
    for attempt in range(1, MAX_RETRY + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            compounds = extract_compounds_from_root(root, prefer_reactions=is_core_metabolism)
            
            # 若 map 模式下请求 rn 图谱为空，则回退请求 map 图谱提取 compound 节点
            if not compounds and species_code == "map" and target_prefix == "rn":
                req_map = urllib.request.Request(f"https://rest.kegg.jp/get/map{pw_num}/kgml", headers=headers)
                with urllib.request.urlopen(req_map, timeout=30) as resp_map:
                    root_map = ET.fromstring(resp_map.read())
                compounds = extract_compounds_from_root(root_map, prefer_reactions=False)
                
            return compounds
        except Exception:
            if attempt < MAX_RETRY:
                time.sleep(1.5)
            else:
                return set()
    return set()

def fetch_kegg_database(species_code="hsa"):
    species_code = str(species_code).strip().lower()
    if species_code == "all":
        for code in ["hsa", "mmu", "rno", "ath", "dre", "eco", "sce", "map"]:
            fetch_kegg_database(code)
        return

    species_name = SPECIES_MAP.get(species_code, f"Custom Organism ({species_code})")
    print(f"⏳ 正在连接 KEGG 官方服务器（{species_name} [{species_code}]）...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    out_filename = f"kegg_{species_code}.csv"

    print("📥 1/2 正在下载该物种的通路目录...")
    pw_names_all = {}
    try:
        req1 = urllib.request.Request(f"https://rest.kegg.jp/list/pathway/{species_code}", headers=headers)
        with urllib.request.urlopen(req1, timeout=30) as response:
            for line in response:
                parts = line.decode('utf-8').strip().split('\t')
                if len(parts) == 2:
                    pw_id = parts[0].replace('path:', '')
                    pw_num = re.sub(r'^[a-z]+', '', pw_id).strip()
                    clean_pw_name = re.sub(r'\s+-\s+[^-]+$', '', parts[1]).strip()
                    pw_names_all[pw_num] = clean_pw_name
    except Exception as e:
        print(f"❌ 无法获取物种 [{species_code}] 的通路列表，请检查物种代码是否正确: {e}")
        sys.exit(1)

    if not pw_names_all:
        print(f"❌ KEGG 未返回物种 [{species_code}] 的任何通路，请确认代码是否有效。")
        sys.exit(1)

    # 筛选目标通路：1) 编号 < 1100 的核心代谢通路；2) 白名单中的生理/信号/转运通路
    target_pathways = []  # 元素为 (pw_num, pw_name, category_str, is_core_metab)
    for pw_num, pw_name in pw_names_all.items():
        try:
            num_val = int(pw_num)
            if num_val < METABOLISM_MAX_NUM:
                target_pathways.append((pw_num, pw_name, "Metabolism", True))
            elif pw_num in METABOLITE_SIGNALING_WHITELIST:
                target_pathways.append((pw_num, pw_name, "Signaling_Physiological", False))
        except ValueError:
            continue

    n_metab = sum(1 for x in target_pathways if x[3])
    n_sig = len(target_pathways) - n_metab
    print(f"   → 已锁定 {n_metab} 条核心生化代谢通路 + {n_sig} 条生理/转运/信号通路（已剔除 01100~01299 全局总览图）")

    total = len(target_pathways)
    print(f"📥 2/2 正在解析 KGML 树，提取小分子化合物节点（共 {total} 条通路）...")

    records = []
    fail_count = 0
    for idx, (pw_num, pw_name, cat_label, is_core) in enumerate(target_pathways):
        compounds = parse_kgml_pathway(species_code, pw_num, is_core, headers)
        
        # 至少包含 2 个化合物才构成有意义的富集通路
        if len(compounds) >= 2:
            records.append({
                "Pathway": pw_name,
                "Compounds": ";".join(sorted(compounds)),
                "Category": cat_label
            })
        else:
            fail_count += 1

        # 礼貌限流：每次请求间隔 0.28s，防止触发 KEGG API 403 封禁
        time.sleep(0.28)

        if (idx + 1) % 10 == 0 or idx + 1 == total:
            print(f"   → 进度: {idx+1}/{total}（成功保留 {len(records)} 条，跳过无化合物/空通路 {fail_count} 条）")

    if not records:
        print("❌ 错误：未能解析到有效化合物，请检查网络连接后重试。")
        sys.exit(1)

    print("💾 正在整理并保存...")
    df = pd.DataFrame(records)
    df.to_csv(out_filename, index=False)
    
    metab_saved = sum(1 for r in records if r["Category"] == "Metabolism")
    sig_saved = sum(1 for r in records if r["Category"] != "Metabolism")
    total_cpds = len(set(c for r in records for c in r["Compounds"].split(";")))
    print(f"✅ {species_name}: 已保存 {len(df)} 条通路（核心代谢 {metab_saved} 条 + 生理/信号/转运 {sig_saved} 条），共覆盖 {total_cpds} 个独立 KEGG 化合物。")

if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "hsa"
    fetch_kegg_database(code)
