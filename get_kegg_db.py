import urllib.request
import pandas as pd
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed

# 独立物种与专项库预设字典（支持在侧边栏单独选择并同步）
SPECIES_MAP = {
    # 1. 哺乳动物与医学模式生物
    "hsa": "Human (Homo sapiens - 人类)",
    "mmu": "Mouse (Mus musculus - 小鼠)",
    "rno": "Rat (Rattus norvegicus - 大鼠)",
    "ssc": "Pig (Sus scrofa - 猪)",
    "bta": "Cow (Bos taurus - 牛)",
    "cfa": "Dog (Canis familiaris - 犬)",
    # 2. 中药材与植物/农作物独立库
    "tcm": "TCM & Medicinal Plants (中药材/药用植物次生与初级全景库)",
    "ath": "Arabidopsis (Arabidopsis thaliana - 拟南芥/模式植物)",
    "osa": "Rice (Oryza sativa - 水稻)",
    "zma": "Maize (Zea mays - 玉米)",
    "gmx": "Soybean (Glycine max - 大豆)",
    "sly": "Tomato (Solanum lycopersicum - 番茄)",
    "vvi": "Grapevine (Vitis vinifera - 酿酒葡萄)",
    # 3. 葡萄酒与发酵食品独立库
    "wine": "Wine & Fermentation (葡萄酒：酿酒葡萄 vvi + 酿酒酵母 sce 专属库)",
    "sce": "Yeast (Saccharomyces cerevisiae - 酿酒酵母)",
    "eco": "E. coli (Escherichia coli K-12 - 大肠杆菌)",
    "bsu": "B. subtilis (Bacillus subtilis - 枯草芽孢杆菌)",
    # 4. 环境毒理与经典模式动物
    "dre": "Zebrafish (Danio rerio - 斑马鱼)",
    "dme": "Fruit fly (Drosophila melanogaster - 果蝇)",
    "cel": "Nematode (Caenorhabditis elegans - 秀丽隐杆线虫)",
    # 5. 通用参考全库
    "map": "General Reference (跨物种全集参考通路)",
}

# 核心生化代谢通路最大编号 (< 01100 为纯正代谢子通路，剔除 01100~01299 全局总览巨型图)
METABOLISM_MAX_NUM = 1100

# 中药材与植物特征性“次生代谢/天然产物生物合成”通路集合
PLANT_SECONDARY_PATHWAYS = {
    "00402": "Benzoxazinoid biosynthesis",
    "00900": "Terpenoid backbone biosynthesis",
    "00901": "Indole alkaloid biosynthesis",
    "00902": "Monoterpenoid biosynthesis",
    "00903": "Limonene and pinene degradation",
    "00904": "Diterpenoid biosynthesis",
    "00905": "Brassinosteroid biosynthesis",
    "00906": "Carotenoid biosynthesis",
    "00908": "Zeatin biosynthesis",
    "00909": "Sesquiterpenoid and triterpenoid biosynthesis",
    "00940": "Phenylpropanoid biosynthesis",
    "00941": "Flavonoid biosynthesis",
    "00942": "Anthocyanin biosynthesis",
    "00943": "Isoflavonoid biosynthesis",
    "00944": "Flavone and flavonol biosynthesis",
    "00945": "Stilbenoid, diarylheptanoid and gingerol biosynthesis",
    "00950": "Isoquinoline alkaloid biosynthesis",
    "00960": "Tropane, piperidine and pyridine alkaloid biosynthesis",
    "00965": "Betalain biosynthesis",
    "00966": "Glucosinolate biosynthesis",
    "00999": "Biosynthesis of various plant secondary metabolites",
}

# 植物/中药库中需排除的工业外源污染物降解及纯微生物抗生素合成通路
NON_PLANT_EXCLUSION_SET = {
    "00311", "00331", "00332", "00333", "00261", "00401", "00404", "00405",
    "00521", "00524", "00525", "00997", "00998", "01051", "01052", "01053",
    "01054", "01055", "01056", "01057", "01059",
    "00361", "00362", "00363", "00364", "00621", "00622", "00623", "00624",
    "00625", "00626", "00627", "00633", "00642", "00643", "00791", "00930",
    "00980", "00982", "00983", "00984"
}

# 代谢组学高价值“生理/转运/信号/细胞过程”通路白名单（富含关键小分子代谢物节点）
METABOLITE_SIGNALING_WHITELIST = {
    "02010",  # ABC transporters (ABC转运蛋白：跨膜转运氨基酸/糖/脂质/次生代谢物)
    "02020",  # Two-component system (双组分系统：微生物与植物环境感应)
    "02060",  # Phosphotransferase system (PTS - 微生物糖转运系统)
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
CONCURRENT_WORKERS = 3  # 遵守 KEGG API 速率限制 (<= 3 requests/sec)

def get_pathway_category(pw_num, is_core_metab):
    """根据通路编号划分三大功能类别"""
    if not is_core_metab:
        return "Signaling_Physiological"
    try:
        n = int(pw_num)
        if (900 <= n < 1100) or pw_num in PLANT_SECONDARY_PATHWAYS:
            return "Secondary_Metabolism"
    except ValueError:
        pass
    return "Metabolism"

def extract_compounds_from_root(root, prefer_reactions=True, union_all=False):
    """从 KGML XML 树中提取 KEGG 化合物 C-number"""
    compounds = set()
    if prefer_reactions or union_all:
        for reaction in root.findall('reaction'):
            for sub in reaction.findall('substrate'):
                for name in sub.get('name', '').split():
                    if name.startswith('cpd:'):
                        compounds.add(name.replace('cpd:', ''))
            for prod in reaction.findall('product'):
                for name in prod.get('name', '').split():
                    if name.startswith('cpd:'):
                        compounds.add(name.replace('cpd:', ''))

    # 对于次生代谢通路、生理/信号通路或无 <reaction> 节点的情况，合并提取 <entry type="compound">
    if union_all or (not compounds):
        for entry in root.findall('entry'):
            if entry.get('type') == 'compound':
                for name in entry.get('name', '').split():
                    if name.startswith('cpd:'):
                        compounds.add(name.replace('cpd:', ''))
    return compounds

def fetch_single_kgml(prefix_code, pw_num, is_core_metabolism, union_all, headers):
    """请求单个前缀的 KGML 并提取化合物集合"""
    url = f"https://rest.kegg.jp/get/{prefix_code}{pw_num}/kgml"
    for attempt in range(1, MAX_RETRY + 1):
        try:
            time.sleep(0.18)
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                xml_data = response.read()
            root = ET.fromstring(xml_data)
            compounds = extract_compounds_from_root(root, prefer_reactions=is_core_metabolism, union_all=union_all)
            if not compounds and prefix_code == "rn":
                req_map = urllib.request.Request(f"https://rest.kegg.jp/get/map{pw_num}/kgml", headers=headers)
                with urllib.request.urlopen(req_map, timeout=30) as resp_map:
                    root_map = ET.fromstring(resp_map.read())
                compounds = extract_compounds_from_root(root_map, prefer_reactions=False, union_all=True)
            return compounds
        except Exception:
            if attempt < MAX_RETRY:
                time.sleep(1.0)
            else:
                return set()
    return set()

def fetch_organism_pathway_list(org_code, headers):
    """获取指定物种代码的通路目录字典 {pw_num: clean_pw_name}"""
    pw_dict = {}
    req = urllib.request.Request(f"https://rest.kegg.jp/list/pathway/{org_code}", headers=headers)
    with urllib.request.urlopen(req, timeout=30) as response:
        for line in response:
            parts = line.decode('utf-8').strip().split('\t')
            if len(parts) == 2:
                pw_id = parts[0].replace('path:', '')
                pw_num = re.sub(r'^[a-z]+', '', pw_id).strip()
                clean_pw_name = re.sub(r'\s+-\s+[^-]+$', '', parts[1]).strip()
                pw_dict[pw_num] = clean_pw_name
    return pw_dict

def build_target_tasks(species_code, headers):
    """
    根据物种代码构建待抓取任务列表:
    返回元素列表: (pw_num, pw_name, cat_label, is_core, prefixes_to_fetch, use_union_all)
    """
    tasks = []

    if species_code == "tcm":
        # 中药材/药用植物独立库：拟南芥 (ath) 初级代谢 + 全套植物天然产物次生代谢 (map009xx)
        ath_pws = fetch_organism_pathway_list("ath", headers)
        combined_pws = dict(ath_pws)
        for sec_num, sec_name in PLANT_SECONDARY_PATHWAYS.items():
            if sec_num not in combined_pws:
                combined_pws[sec_num] = sec_name

        for pw_num, pw_name in sorted(combined_pws.items()):
            if pw_num in NON_PLANT_EXCLUSION_SET:
                continue
            try:
                num_val = int(pw_num)
                if num_val < METABOLISM_MAX_NUM:
                    cat_label = get_pathway_category(pw_num, True)
                    is_sec = (cat_label == "Secondary_Metabolism")
                    # 次生代谢通路直接从标准参考图提取全化合物节点，初级代谢从 ath 提取
                    prefixes = ["map"] if (is_sec or pw_num not in ath_pws) else ["ath"]
                    tasks.append((pw_num, pw_name, cat_label, True, prefixes, is_sec))
                elif pw_num in {"02010", "04070", "04072", "04075"}:
                    tasks.append((pw_num, pw_name, "Signaling_Physiological", False, ["ath"], True))
            except ValueError:
                continue

    elif species_code == "wine":
        # 葡萄酒独立库：酿酒葡萄 (vvi) + 酿酒酵母 (sce) 双界合并
        vvi_pws = fetch_organism_pathway_list("vvi", headers)
        sce_pws = fetch_organism_pathway_list("sce", headers)
        all_nums = sorted(set(vvi_pws.keys()).union(set(sce_pws.keys())))

        for pw_num in all_nums:
            if pw_num in NON_PLANT_EXCLUSION_SET:
                continue
            pw_name = vvi_pws.get(pw_num) or sce_pws.get(pw_num)
            try:
                num_val = int(pw_num)
                if num_val < METABOLISM_MAX_NUM:
                    cat_label = get_pathway_category(pw_num, True)
                    is_sec = (cat_label == "Secondary_Metabolism")
                    prefixes = []
                    if pw_num in vvi_pws: prefixes.append("vvi")
                    if pw_num in sce_pws and not is_sec: prefixes.append("sce")
                    tasks.append((pw_num, pw_name, cat_label, True, prefixes, is_sec))
                elif pw_num in {"02010", "04020", "04024", "04070", "04075", "04150", "04152"}:
                    prefixes = ["vvi"] if pw_num in vvi_pws else ["sce"]
                    tasks.append((pw_num, pw_name, "Signaling_Physiological", False, prefixes, True))
            except ValueError:
                continue

    else:
        # 常规单一物种或 map 通用全库
        org_pws = fetch_organism_pathway_list(species_code, headers)
        for pw_num, pw_name in sorted(org_pws.items()):
            try:
                num_val = int(pw_num)
                if num_val < METABOLISM_MAX_NUM:
                    cat_label = get_pathway_category(pw_num, True)
                    is_sec = (cat_label == "Secondary_Metabolism")
                    pref = "rn" if species_code == "map" else species_code
                    tasks.append((pw_num, pw_name, cat_label, True, [pref], is_sec))
                elif pw_num in METABOLITE_SIGNALING_WHITELIST:
                    pref = "map" if species_code == "map" else species_code
                    tasks.append((pw_num, pw_name, "Signaling_Physiological", False, [pref], True))
            except ValueError:
                continue

    return tasks

def worker_run_task(task_tuple, headers):
    """线程工作函数：抓取单条通路下的所有前缀并求化合物并集"""
    pw_num, pw_name, cat_label, is_core, prefixes, use_union_all = task_tuple
    merged_compounds = set()
    for pref in prefixes:
        cpds = fetch_single_kgml(pref, pw_num, is_core, use_union_all, headers)
        merged_compounds.update(cpds)
    # 若植物次生代谢通路在特定物种下仅注释了极少反应，自动补齐 map 标准次生代谢物节点
    if cat_label == "Secondary_Metabolism" and len(merged_compounds) < 5 and "map" not in prefixes:
        cpds_map = fetch_single_kgml("map", pw_num, False, True, headers)
        merged_compounds.update(cpds_map)
    return pw_num, pw_name, cat_label, merged_compounds

def fetch_kegg_database(species_code="hsa"):
    species_code = str(species_code).strip().lower()
    if species_code == "all":
        for code in ["hsa", "mmu", "rno", "tcm", "wine", "ath", "dre", "eco", "sce", "map"]:
            fetch_kegg_database(code)
        return

    species_name = SPECIES_MAP.get(species_code, f"Custom Organism ({species_code})")
    print(f"⏳ 正在连接 KEGG 官方服务器（{species_name} [{species_code}]）...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    out_filename = f"kegg_{species_code}.csv"

    print("📥 1/2 正在构建通路目录...")
    try:
        target_tasks = build_target_tasks(species_code, headers)
    except Exception as e:
        print(f"❌ 无法获取 [{species_code}] 的通路目录，请检查物种代码或网络连接: {e}")
        sys.exit(1)

    if not target_tasks:
        print(f"❌ KEGG 未返回 [{species_code}] 的有效通路，请确认代码是否正确。")
        sys.exit(1)

    n_prim = sum(1 for t in target_tasks if t[2] == "Metabolism")
    n_sec = sum(1 for t in target_tasks if t[2] == "Secondary_Metabolism")
    n_sig = sum(1 for t in target_tasks if t[2] == "Signaling_Physiological")
    total = len(target_tasks)
    print(f"   → 已锁定 {total} 条通路（初级代谢 {n_prim} + 次生代谢/天然产物 {n_sec} + 生理/信号/转运 {n_sig}）")
    print(f"📥 2/2 正在启动 {CONCURRENT_WORKERS} 线程并发解析 KGML 化合物节点...")

    records = []
    fail_count = 0
    completed_count = 0

    with ThreadPoolExecutor(max_workers=CONCURRENT_WORKERS) as executor:
        future_to_task = {executor.submit(worker_run_task, task, headers): task for task in target_tasks}
        for future in as_completed(future_to_task):
            completed_count += 1
            try:
                pw_num, pw_name, cat_label, compounds = future.result()
                if len(compounds) >= 2:
                    records.append({
                        "__num__": pw_num,
                        "Pathway": pw_name,
                        "Compounds": ";".join(sorted(compounds)),
                        "Category": cat_label
                    })
                else:
                    fail_count += 1
            except Exception:
                fail_count += 1

            if completed_count % 15 == 0 or completed_count == total:
                print(f"   → 进度: {completed_count}/{total}（已成功解析 {len(records)} 条，跳过空通路 {fail_count} 条）")

    if not records:
        print("❌ 错误：未能解析到有效化合物，请检查网络连接后重试。")
        sys.exit(1)

    records.sort(key=lambda r: r["__num__"])
    df = pd.DataFrame(records).drop(columns=["__num__"])
    df.to_csv(out_filename, index=False)
    
    prim_saved = sum(1 for r in records if r["Category"] == "Metabolism")
    sec_saved = sum(1 for r in records if r["Category"] == "Secondary_Metabolism")
    sig_saved = sum(1 for r in records if r["Category"] == "Signaling_Physiological")
    total_cpds = len(set(c for r in records for c in r["Compounds"].split(";")))
    print(f"✅ {species_name}: 已保存 {len(df)} 条通路（初级代谢 {prim_saved} + 次生代谢 {sec_saved} + 生理/信号 {sig_saved}），覆盖 {total_cpds} 个唯一化合物。")

if __name__ == "__main__":
    code = sys.argv[1] if len(sys.argv) > 1 else "hsa"
    fetch_kegg_database(code)
