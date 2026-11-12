import streamlit as st
import pandas as pd
import numpy as np
import os
import gc
import datetime
import re
import io
import base64
import subprocess
import sys
import csv 
import traceback
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
from sklearn.decomposition import PCA

# ==========================================
# 0. 品牌更新与 UI 配置
# ==========================================
st.set_page_config(page_title="MetaFlow Studio", page_icon="🧬", layout="wide")

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

try:
    from data_preprocessing import (
        data_cleaning_pipeline, 
        parse_metdna_file, 
        parse_universal_single_table, 
        parse_manual_targeted_files,
        merge_multiple_dfs, 
        align_sample_info, 
        make_unique,
        OPLS_DA, 
        run_pathway_enrichment, 
        build_kegg_dictionary
    )
    from stats_utils import run_pairwise_statistics
    from plot_utils import update_layout_square, get_ellipse_coordinates, plot_nomogram
    from report_generator import generate_offline_html, generate_ai_prompt
except ImportError as e:
    st.error(f"❌ 严重错误：未找到核心依赖文件。详情: {e}")
    st.stop()

has_serrf = False
try:
    from serrf_module import serrf_normalization
    has_serrf = True
except ImportError:
    pass

COLOR_PALETTE = {'Up': '#CD0000', 'Down': '#00008B', 'NS': '#E0E0E0'} 
GROUP_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
MODE_COLORS = ['#2b5c8f', '#d95f02', '#7570b3', '#1b9e77', '#e7298a', '#66a61e']
MODE_SYMBOLS = ['circle', 'square', 'diamond', 'cross', 'triangle-up', 'star']

st.markdown("""
<style>
    .block-container {padding-top: 2rem !important; padding-bottom: 3rem !important;}
    h1, h2, h3, div, p {font-family: 'Arial', sans-serif; color: #2c3e50;}
    div[data-testid="stForm"] button {width: 100%; background-color: #ff4b4b; color: white; font-weight: bold;}
    .logo-container {display: flex; align-items: center; margin-bottom: 20px;}
    .logo-text {font-size: 2.5rem; font-weight: 800; color: #2c3e50; margin-left: 15px; letter-spacing: -1px;}
    .logo-badge {background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); color: white; padding: 4px 12px; border-radius: 20px; font-size: 1rem; margin-left: 10px; vertical-align: middle;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="logo-container">
    <span style="font-size: 3rem;">♾️</span>
    <span class="logo-text">MetaFlow Studio</span>
    <span class="logo-badge">Pro</span>
</div>
""", unsafe_allow_html=True)

# ==========================================
# 1. 极简侧边栏数据流控制台
# ==========================================
if 'raw_df' not in st.session_state: st.session_state.raw_df = None
if 'feature_meta' not in st.session_state: st.session_state.feature_meta = None
if 'mode_datasets' not in st.session_state: st.session_state.mode_datasets = []
if 'data_loaded' not in st.session_state: st.session_state.data_loaded = False

DEFAULT_4_MODES = ["RP+", "RP-", "HILIC+", "HILIC-"]

def infer_default_mode_name(filename, idx):
    """根据文件名智能推断模式名，若文件名完全相同则按 RP+, RP-, HILIC+, HILIC- 顺序赋予默认值"""
    fn_upper = os.path.splitext(filename)[0].upper()
    if "HILIC" in fn_upper and ("POS" in fn_upper or "+" in fn_upper): return "HILIC+"
    if "HILIC" in fn_upper and ("NEG" in fn_upper or "-" in fn_upper): return "HILIC-"
    if "RP" in fn_upper and ("POS" in fn_upper or "+" in fn_upper): return "RP+"
    if "RP" in fn_upper and ("NEG" in fn_upper or "-" in fn_upper): return "RP-"
    if idx < len(DEFAULT_4_MODES): return DEFAULT_4_MODES[idx]
    return f"Mode_{idx+1}"

def safe_filename_part(s):
    """清理文件名中的非法字符，防止浏览器下载时截断后缀变成 .txt"""
    cleaned = re.sub(r'[^a-zA-Z0-9_\-\u4e00-\u9fa5]', '_', str(s).strip())
    return cleaned if cleaned else "Group"

def format_hm_row_label(name_str, mode_str, max_chem_len=46):
    """格式化热图行标签：对超长化学名做优雅截断，确保末尾 [模式] 标识绝不被裁切"""
    clean_n = str(name_str).strip()
    clean_m = str(mode_str).strip()
    if clean_m and clean_n.endswith(f"[{clean_m}]"):
        clean_n = clean_n[:-len(f"[{clean_m}]")].strip()
    if len(clean_n) > max_chem_len:
        clean_n = clean_n[:max_chem_len - 3].rstrip() + "..."
    return f"{clean_n} [{clean_m}]" if clean_m else clean_n

with st.sidebar:
    st.header("🛠️ 数据控制台")
    
    data_source = st.radio(
        "选择数据流模式", 
        ["1. MA 标准单表 (自带分组)", "2. 拟靶向 MRM 宽表 (需后缀)", "3. MetDNA 原始宽表"], 
        index=0
    )
    
    info_df = None; candidate_samples = []; user_sample_col = None; user_group_col = None
    excluded_samples = []
    use_serrf = False; serrf_ready = False
    run_order_col = None; sample_type_col = None; qc_label = "QC"
    
    if data_source in ["2. 拟靶向 MRM 宽表 (需后缀)", "3. MetDNA 原始宽表"]:
        st.markdown("#### 1. 上传 Sample Info")
        sample_info_file = st.file_uploader("Info表格 (.csv/.xlsx) [必填]", type=["csv", "xlsx"], key="info")
        if sample_info_file:
            try:
                sample_info_file.seek(0)
                info_df = pd.read_csv(sample_info_file) if sample_info_file.name.endswith('.csv') else pd.read_excel(sample_info_file)
                cols = list(info_df.columns); cols_lower = [c.lower() for c in cols]
                idx_sample = next((cols_lower.index(kw) for kw in ['sample.name', 'sample_name', 'sample', 'name', 'id'] if kw in cols_lower), 0)
                idx_group = next((cols_lower.index(kw) for kw in ['group', 'class', 'type', 'condition'] if kw in cols_lower), 1 if len(cols) > 1 else 0)
                c1, c2 = st.columns(2)
                user_sample_col = c1.selectbox("样本列", cols, index=idx_sample)
                user_group_col = c2.selectbox("分组列", cols, index=idx_group)
                if user_sample_col: candidate_samples = info_df[user_sample_col].astype(str).unique().tolist()
            except Exception as e: st.error(f"Info 读取失败: {e}")
            
        excluded_samples = st.multiselect("2. 样本剔除 (黑名单)", options=candidate_samples, default=[])
        
        use_serrf = st.checkbox("3. 启用 SERRF 批次校正 (按单表独立执行)", value=False)
        if use_serrf:
            if not has_serrf: st.warning("⚠️ 缺失 serrf_module.py")
            elif info_df is not None:
                idx_order = next((i for i, c in enumerate(cols_lower) if any(x in c for x in ['order', 'run', 'idx', 'seq'])), 0)
                sc1, sc2, sc3 = st.columns(3)
                run_order_col = sc1.selectbox("进样顺序列", cols, index=idx_order)
                sample_type_col = sc2.selectbox("样本类型列", cols, index=cols.index(user_group_col) if user_group_col else 0)
                default_qc = next((v for v in info_df[sample_type_col].unique().astype(str) if 'qc' in v.lower()), "QC")
                qc_label = sc3.text_input("QC标识", value=default_qc)
                serrf_ready = True
    else:
        st.markdown("#### 1. 上传数据矩阵")
        ex_str = st.text_input("2. 样本剔除 (选填，逗号分隔)")
        if ex_str: excluded_samples = [s.strip() for s in ex_str.split(',') if s.strip()]

    st.markdown("#### 3. KEGG 通路配置")
    species = st.selectbox("物种背景", ["Human (Homo sapiens)", "Mouse (Mus musculus)", "Rat (Rattus norvegicus)", "General (所有物种)"], index=0)
    species_code = {"Human (Homo sapiens)": "hsa", "Mouse (Mus musculus)": "mmu", "Rat (Rattus norvegicus)": "rno", "General (所有物种)": "map"}[species]
    db_filename = f"kegg_{species_code}.csv"
    custom_pathway_file = st.file_uploader("自定义通路库 (.csv)", type=["csv", "gmt"], key="pathway_db")
    
    db_sync_msg = st.session_state.get('_db_sync_msg', '')
    if st.button(f"🔄 同步 {species_code} 通路库", width='stretch'):
        with st.spinner(f"⏳ 正在后台解析 KGML 抓取 {species_code} 核心代谢通路（约需 1-2 分钟）..."):
            try:
                res_sub = subprocess.run([sys.executable, "get_kegg_db.py", species_code], capture_output=True, text=True)
                if res_sub.returncode == 0:
                    st.session_state['_db_sync_msg'] = f"✅ {species_code} 库同步成功！\n*(更新时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})*"
                    st.toast("✅ 库同步成功！")
                    st.rerun() 
                else:
                    st.session_state['_db_sync_msg'] = f"❌ 更新脚本执行报错:\n```text\n{res_sub.stderr or res_sub.stdout}\n```"
            except Exception as e:
                st.session_state['_db_sync_msg'] = f"❌ 无法启动更新程序: {str(e)}"
                
    if db_sync_msg:
        if db_sync_msg.startswith('✅'): st.success(db_sync_msg)
        else: st.error(db_sync_msg)
        
    if os.path.exists(db_filename):
        try:
            _kegg_df = pd.read_csv(db_filename)
            _n_pws = len(_kegg_df)
            _n_cpds = sum(len(str(c).split(';')) for c in _kegg_df['Compounds'].dropna())
            st.caption(f"📦 当前库：`{db_filename}`（{_n_pws} 条通路 | {_n_cpds} 个物种特异反应节点）")
        except Exception:
            st.caption(f"📦 当前库：`{db_filename}`")
    else:
        st.caption(f"⚠️ 未找到 `{db_filename}`，请点击上方同步按钮。")

    st.markdown("#### 4. 上传分析数据")
    feature_scope = "全部特征"
    dedup_within_file = True
    dict_files = None
    custom_mode_tags = []
    
    if data_source == "1. MA 标准单表 (自带分组)":
        dict_files = st.file_uploader("关联 MetDNA 字典 (可选)", type=["csv", "xlsx"], accept_multiple_files=True, key="dict_files")
        uploaded_files = st.file_uploader("上传 MA 格式单表", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    elif data_source == "2. 拟靶向 MRM 宽表 (需后缀)":
        suffix = st.text_input("提取指标后缀", value=" : 面积")
        dict_files = st.file_uploader("关联 MetDNA 字典 (可选)", type=["csv", "xlsx"], accept_multiple_files=True, key="dict_files")
        uploaded_files = st.file_uploader("上传 MRM 宽表", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
    else:
        feature_scope = st.radio("特征范围", ["仅已注释特征", "全部特征"], index=0)
        dedup_within_file = st.checkbox(
            "单模式内同名化合物去重 (合并同分异构体/加合离子)",
            value=True,
            help="开启后，单个模式文件内注释为同一名称的多个色谱峰将按三级漏斗保留最优峰；若需与未去重的原始峰表直接对比特征数，可取消勾选。"
        )
        uploaded_files = st.file_uploader("上传 MetDNA 结果表 (支持 4 模式同时上传)", type=["csv", "xlsx"], accept_multiple_files=True, key="data")
        if uploaded_files:
            st.markdown("##### 📌 采集模式标识 (用于图表与清单溯源)")
            st.caption("即使上传的 4 个表格文件名完全相同，系统也会按以下模式名严格区分：")
            cols_m = st.columns(2)
            raw_tags = []
            for idx, f_obj in enumerate(uploaded_files):
                default_tag = infer_default_mode_name(f_obj.name, idx)
                with cols_m[idx % 2]:
                    t_val = st.text_input(f"文件 {idx+1} ({f_obj.name[:12]})", value=default_tag, key=f"mode_tag_{idx}")
                    raw_tags.append(t_val.strip() if t_val.strip() else f"Mode_{idx+1}")
            custom_mode_tags = make_unique(raw_tags)
        
    start_process = st.container().button("📥 加载数据矩阵", width='stretch', type="primary")


def apply_single_table_qc_and_serrf(df_t, meta, mode_tag, info_df, user_sample_col, user_group_col, excluded_samples, use_serrf, serrf_ready, run_order_col, sample_type_col, qc_label):
    """辅助函数：在单个模式表格内部独立完成分组对齐、样本黑名单剔除与 SERRF 批次校正"""
    if info_df is not None:
        info_aligned = align_sample_info(df_t, info_df, sample_col_name=user_sample_col)
        if user_group_col and user_group_col in info_aligned.columns:
            df_t['Group'] = info_aligned[user_group_col].fillna('Unknown').values

    if excluded_samples:
        ex_fps = set([re.sub(r'[^a-z0-9]', '', str(s).strip().lower()) for s in excluded_samples])
        df_t = df_t[~df_t['SampleID'].astype(str).apply(lambda s: re.sub(r'[^a-z0-9]', '', str(s).strip().lower())).isin(ex_fps)].reset_index(drop=True)

    if use_serrf and serrf_ready and has_serrf and info_df is not None:
        try:
            aligned_info = align_sample_info(df_t, info_df, sample_col_name=user_sample_col)
            num_cols = [c for c in df_t.columns if c not in ['SampleID', 'Group', 'Source_Files']]
            df_data_in = df_t[num_cols].copy()
            df_corrected, rsd_info = serrf_normalization(df_data_in, aligned_info, run_order_col, sample_type_col, qc_label)
            if df_corrected is not None:
                df_t = df_t.loc[df_corrected.index].copy()
                df_t.loc[:, num_cols] = df_corrected.values
                if meta is not None and 'Mean_Area' in meta.columns:
                    meta.loc[num_cols, 'Mean_Area'] = df_t[num_cols].mean(axis=0).values
                st.toast(f"✅ [{mode_tag}] SERRF 校正完成: QC 中位 RSD 从 {rsd_info['RSD_Before']:.1f}% 降至 {rsd_info['RSD_After']:.1f}%")
            else:
                st.warning(f"⚠️ [{mode_tag}] SERRF 跳过: {rsd_info}")
        except Exception as e:
            st.warning(f"⚠️ [{mode_tag}] SERRF 异常 ({str(e)})，已保留未校正数据。")
    return df_t, meta

# ==========================================
# 2. 核心路由与解析引擎调用
# ==========================================
if start_process:
    if 'analysis_res' in st.session_state: del st.session_state['analysis_res']
        
    if not uploaded_files: 
        st.error("请先上传主数据文件！")
    else:
        progress_bar = st.progress(0)
        final_df = None; final_meta = None; mode_datasets = []
        
        if data_source == "1. MA 标准单表 (自带分组)":
            with st.spinner("启动 MA 引擎..."):
                ext_dict = build_kegg_dictionary(dict_files) if dict_files else {}
                df_t, meta, err = parse_universal_single_table(uploaded_files, external_kegg_dict=ext_dict)
                if err: st.error(err)
                else:
                    df_t, meta = apply_single_table_qc_and_serrf(df_t, meta, "MA_Single", None, None, None, excluded_samples, False, False, None, None, None)
                    final_df, final_meta = df_t, meta
                    mode_datasets = [(df_t, meta, "MA_Single")]

        elif data_source == "2. 拟靶向 MRM 宽表 (需后缀)":
            if info_df is None: st.error("⚠️ 请先上传 Info 表格！")
            else:
                with st.spinner("启动 MRM 引擎..."):
                    ext_dict = build_kegg_dictionary(dict_files) if dict_files else {}
                    df_t, meta, err = parse_manual_targeted_files(uploaded_files, metric_suffix=suffix, external_kegg_dict=ext_dict)
                    if err: st.error(err)
                    else:
                        df_t, meta = apply_single_table_qc_and_serrf(df_t, meta, "MRM_Targeted", info_df, user_sample_col, user_group_col, excluded_samples, use_serrf, serrf_ready, run_order_col, sample_type_col, qc_label)
                        final_df, final_meta = df_t, meta
                        mode_datasets = [(df_t, meta, "MRM_Targeted")]

        else:
            if info_df is None: 
                st.error("⚠️ 请先上传 Info 表格！")
            else:
                with st.spinner("启动 MetDNA 分模式独立质控与三级漏斗解析引擎..."):
                    parsed_results = []
                    for i, file in enumerate(uploaded_files):
                        assigned_tag = custom_mode_tags[i] if i < len(custom_mode_tags) else infer_default_mode_name(file.name, i)
                        df_t, meta, err = parse_metdna_file(
                            file, file.name, valid_samples=candidate_samples,
                            dedup_within_file=dedup_within_file, custom_mode_tag=assigned_tag
                        )
                        
                        if not err:
                            if feature_scope.startswith("仅已注释"):
                                anno_ids = set(meta[meta['Is_Annotated'] == True].index.astype(str).tolist())
                                keep_cols = ['SampleID', 'Group', 'Source_Files'] + [c for c in df_t.columns if str(c) in anno_ids]
                                df_t = df_t[keep_cols].copy()
                                meta = meta.loc[meta.index.astype(str).isin([str(c) for c in df_t.columns])].copy()
                            
                            df_t, meta = apply_single_table_qc_and_serrf(
                                df_t, meta, assigned_tag, info_df, user_sample_col, user_group_col,
                                excluded_samples, use_serrf, serrf_ready, run_order_col, sample_type_col, qc_label
                            )
                            parsed_results.append((df_t, meta, assigned_tag))
                        else:
                            st.error(f"❌ 文件 {file.name} ({assigned_tag}) 解析失败: {err}")
                            
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if parsed_results:
                        raw_df, merged_meta, err = merge_multiple_dfs(parsed_results)
                        if err: st.error(err)
                        else:
                            info_aligned = align_sample_info(raw_df, info_df, sample_col_name=user_sample_col)
                            if user_group_col and user_group_col in info_aligned.columns: 
                                raw_df['Group'] = info_aligned[user_group_col].fillna('Unknown').values
                            final_df, final_meta = raw_df, merged_meta
                            mode_datasets = parsed_results
                    else:
                        st.error("⚠️ 未能提取到有效数据，请检查 Info 表格式！")

        if final_df is not None:
            st.session_state.raw_df = final_df
            st.session_state.feature_meta = final_meta
            st.session_state.mode_datasets = mode_datasets
            st.session_state.data_loaded = True
            st.rerun()

# ==========================================
# 3. 统计与可视化参数设置 (全面对齐 MetaboAnalyst 标准)
# ==========================================
if not st.session_state.data_loaded:
    st.info("👈 请在左侧控制台按对应数据格式上传数据。")
    st.stop()

raw_df = st.session_state.raw_df
mode_datasets = st.session_state.get('mode_datasets', [])
mode_count = len(mode_datasets)

if mode_count > 1:
    mode_summary_str = " &nbsp;|&nbsp; ".join([f"**`{m_name}`** ({len(m_df.columns)-3} 特征)" for m_df, _, m_name in mode_datasets])
    st.info(f"📊 **多模式数据总览**: 已成功加载 **{mode_count}** 个独立采集模式 [ {mode_summary_str} ] &nbsp;➔&nbsp; 跨模式合并去重后共 **{len(raw_df)}** 样本 x **{len(raw_df.columns)-3}** 唯一代谢物")
else:
    st.info(f"📊 **数据总览**: {len(raw_df)} 样本 x {len(raw_df.columns)-3} 特征")

st.download_button("📥 导出跨模式合并矩阵 (.csv)", raw_df.to_csv(index=False).encode('utf-8-sig'), "MetaFlow_Merged_Matrix.csv", "text/csv")
st.divider()

with st.form(key='analysis_form'):
    st.markdown("### ⚙️ 分析参数配置")
    non_num = raw_df.select_dtypes(exclude=[np.number]).columns.tolist()
    group_col = st.selectbox("分组列", non_num, index=non_num.index('Group') if 'Group' in non_num else 0)
    
    c_sw1, c_sw2 = st.columns(2)
    with c_sw1:
        integration_mode = st.radio(
            "🔀 多模式数据整合与统计策略",
            [
                "模式 A：单模式独立归一化 ➔ 跨模式优选去重 ➔ 全局统一建模 (推荐临床/机器学习)",
                "模式 B：各模式独立统计建模 ➔ 差异代谢物去重合并 ➔ 通路富集 (JHM 经典非靶向流派)"
            ],
            index=0,
            disabled=(mode_count <= 1),
            help="当上传多个模式文件（如 RP+, RP-, HILIC+, HILIC-）时，可随时切换两种学术界主流范式，无需重新上传数据。"
        )
    with c_sw2:
        kegg_bg_option = st.radio(
            "🎯 KEGG 超几何分布富集背景 (Background Universe)",
            [
                "物种全库参考背景 (同 MetaboAnalyst 默认，高灵敏度)",
                "本次实验实测背景 (严格模式，仅以检出代谢物为分母 N)"
            ],
            index=0,
            help="物种全库参考背景以当前物种 KEGG 代谢网络全集为 N；实验实测背景以本次上传表格中检出的 KEGG 代谢物为 N。"
        )
    
    with st.expander("数据预处理与统计检验配置 (已对齐 MetaboAnalyst 默认标准)", expanded=True):
        c_p1, c_p2, c_p3 = st.columns(3)
        miss_th = c_p1.slider(
            "缺失率过滤阈值 (MetaboAnalyst 默认为 0.50)",
            0.0, 1.0, 0.50, step=0.05,
            help="仅在所选的两个对比组样本中计算缺失率，无关组不会连坐误删特征。"
        )
        miss_filter_rule = c_p2.selectbox(
            "缺失率评估规则",
            [
                "组内修正过滤 (任一对比组内缺失率 ≤ 阈值即保留，推荐)",
                "对比组全局过滤 (同 MetaboAnalyst 默认)"
            ],
            index=0,
            help="『组内修正过滤』可完美保留在某一组 100% 检出、另一组未检出的全有/全无型特异生物标志物。"
        )
        impute_label = c_p3.selectbox(
            "缺失值填充方法",
            [
                "1/5 Min (LoD, 同 MetaboAnalyst 默认)",
                "1/2 Min",
                "KNN (K近邻插补)",
                "mean (均值)",
                "zero (极小常数)"
            ],
            index=0
        )
        
        c_p4, c_p5, c_p6 = st.columns(3)
        norm_m = c_p4.selectbox("样本归一化 (单模式内独立执行)", ["PQN", "Median", "Sum", "None"], index=0)
        scale_m = c_p5.selectbox("多变量特征缩放", ["Pareto", "Auto", "None"], index=0)
        ttest_var_label = c_p6.selectbox(
            "T 检验方差假设",
            [
                "Student's t-test (方差齐性, 同 MetaboAnalyst 默认)",
                "Welch's t-test (方差不齐)"
            ],
            index=0
        )
        do_log = st.checkbox("Log2 对数转化 (Log2_FC 严格在归一化线性尺度计算算术均值比值，与 MetaboAnalyst 一致)", value=True)

    with st.expander("图表设置", expanded=False):
        c_t1, c_t2, c_t3 = st.columns(3)
        vip_show_num = c_t1.slider("VIP 展示数", 10, 50, 25)
        nomo_num = c_t2.slider("列线图标志物数", 2, 8, 4)
        pw_show_num = c_t3.slider("网络图通路数", 5, 30, 15)

    cur_grps = sorted(raw_df[group_col].astype(str).unique())
    default_grps = [g for g in cur_grps if 'qc' not in g.lower()][:2]
    if len(default_grps) < 2: default_grps = cur_grps[:2]
    sel_grps = st.multiselect("选择对比组 (限 2 组)", cur_grps, default=default_grps)
    c1, c2, c3, c4, c5 = st.columns(5)
    case = c1.selectbox("Case 组 (分子)", list(sel_grps), index=0 if sel_grps else None)
    ctrl = c2.selectbox("Control 组 (分母)", list(sel_grps), index=1 if len(sel_grps)>1 else 0)
    p_th = c3.number_input("P-value 阈值", value=0.05, step=0.01)
    fc_th = c4.number_input("Log2 FC 阈值", value=0.58, step=0.10, help="0.58 对应 1.5 倍变化，1.0 对应 2.0 倍变化")
    vip_th = c5.number_input("VIP 阈值 (设0仅看P与FC)", value=1.0, min_value=0.0, step=0.10, help="默认 1.0；若设为 0.0 则与 MetaboAnalyst 单变量火山图筛选标准完全一致")
    submit_button = st.form_submit_button(label='🚀 执行分析')


# ==========================================
# 辅助函数 1：构建带模式形状与可选化合物标签的火山图
# ==========================================
def build_volcano_figure(stats_df, p_th, fc_th, title="Volcano Plot", show_labels=False, top_label_n=10):
    plot_df = stats_df.copy()
    if 'Source_Mode' not in plot_df.columns:
        plot_df['Source_Mode'] = 'Global'
    if 'Detected_Modes' not in plot_df.columns:
        plot_df['Detected_Modes'] = plot_df['Source_Mode']
        
    plot_df['Name_With_Mode'] = plot_df['Name'].astype(str) + " [" + plot_df['Source_Mode'].astype(str) + "]"
    
    unique_modes = plot_df['Source_Mode'].astype(str).unique().tolist()
    if len(unique_modes) > 1:
        fig_vol = px.scatter(
            plot_df, x="Log2_FC", y="-Log10_P", color="Sig", symbol="Source_Mode",
            color_discrete_map=COLOR_PALETTE, symbol_sequence=MODE_SYMBOLS,
            hover_name="Name_With_Mode",
            hover_data={'Source_Mode': True, 'Detected_Modes': True, 'VIP': ':.2f', 'Log2_FC': ':.2f', 'P_Value': ':.2e', 'FDR': ':.2e'}
        )
    else:
        fig_vol = px.scatter(
            plot_df, x="Log2_FC", y="-Log10_P", color="Sig",
            color_discrete_map=COLOR_PALETTE,
            hover_name="Name_With_Mode",
            hover_data={'Source_Mode': True, 'VIP': ':.2f', 'Log2_FC': ':.2f', 'P_Value': ':.2e', 'FDR': ':.2e'}
        )
        
    fig_vol.update_traces(marker=dict(size=9, opacity=0.85, line=dict(width=0.5, color='white')))
    fig_vol.add_hline(y=-np.log10(p_th), line_dash="dash", line_color="gray")
    fig_vol.add_vline(x=fc_th, line_dash="dash", line_color="gray")
    fig_vol.add_vline(x=-fc_th, line_dash="dash", line_color="gray")

    if show_labels and top_label_n > 0:
        if 'Is_Biomarker' in plot_df.columns:
            sig_sub = plot_df[plot_df['Is_Biomarker'] == True].copy()
        else:
            sig_sub = plot_df[plot_df['Sig'] != 'NS'].copy()
        if sig_sub.empty:
            sig_sub = plot_df[plot_df['Sig'] != 'NS'].copy()
        if not sig_sub.empty:
            top_annot = sig_sub.sort_values(['P_Value', 'VIP'], ascending=[True, False]).head(int(top_label_n))
            for idx_r, (_, r) in enumerate(top_annot.iterrows()):
                ay_offset = -22 if idx_r % 2 == 0 else 22
                fig_vol.add_annotation(
                    x=float(r['Log2_FC']), y=float(r['-Log10_P']),
                    text=f"{r['Name']} [{r['Source_Mode']}]",
                    showarrow=True, arrowhead=1, arrowsize=0.8, arrowwidth=1,
                    arrowcolor="#555", ax=0, ay=ay_offset,
                    font=dict(size=10, color="#222"),
                    bgcolor="rgba(255,255,255,0.82)"
                )

    fig_vol = update_layout_square(fig_vol, title, "Log2 FC", "-Log10(P)")
    fig_vol.update_layout(width=720, height=650)
    return fig_vol


# ==========================================
# 辅助函数 2：出版级全功能热图生成器 (零重叠精准坐标布局)
# ==========================================
def generate_custom_heatmap(
    df_all_unscaled, out_df, group_col, sel_grps,
    scope_mode="pairwise", hm_type="sample_sorted",
    top_n=40, show_row_mode=True, show_cell_values=False
):
    if out_df is None or out_df.empty:
        return None, None, "当前筛选阈值下无显著差异代谢物。"

    valid_mets = [m for m in out_df['Metabolite'].tolist() if m in df_all_unscaled.columns]
    if not valid_mets:
        return None, None, "未在丰度矩阵中匹配到差异代谢物特征列。"

    hm_feats = valid_mets[:min(int(top_n), len(valid_mets))]
    sub_out = out_df.set_index('Metabolite').loc[hm_feats]
    raw_names = sub_out['Name'].astype(str).tolist()
    modes_list = sub_out.get('Source_Mode', pd.Series(['Global']*len(hm_feats), index=hm_feats)).astype(str).tolist()
    display_names = make_unique([format_hm_row_label(n, m, max_chem_len=48) for n, m in zip(raw_names, modes_list)])

    df_scope = df_all_unscaled.copy()
    if scope_mode == "pairwise":
        df_scope = df_scope[df_scope[group_col].isin(sel_grps)].copy()
    elif scope_mode == "all_no_qc":
        df_scope = df_scope[~df_scope[group_col].astype(str).str.lower().str.contains('qc')].copy()

    if len(df_scope) < 2:
        return None, None, "所选样本范围内有效样本数不足 2 个，无法绘制热图。"

    all_present_grps = df_scope[group_col].astype(str).unique().tolist()
    ordered_grps = [g for g in sel_grps if g in all_present_grps] + sorted([g for g in all_present_grps if g not in sel_grps])
    df_scope['__grp_order__'] = df_scope[group_col].astype(str).map({g: idx for idx, g in enumerate(ordered_grps)})
    df_scope = df_scope.sort_values(by=['__grp_order__', 'SampleID']).drop(columns=['__grp_order__']).reset_index(drop=True)

    unique_modes = list(dict.fromkeys(modes_list))
    mode_lut = {m: MODE_COLORS[i % len(MODE_COLORS)] for i, m in enumerate(unique_modes)}
    row_colors_series = pd.Series([mode_lut[m] for m in modes_list], index=display_names) if (show_row_mode and len(unique_modes) > 1) else None

    plt.close('all')
    try:
        do_row_cluster = (len(hm_feats) >= 2)
        do_col_cluster = False
        legend_handles = []
        cbar_title = "Z-Score"

        if hm_type == "correlation":
            if len(hm_feats) < 2:
                return None, None, "至少需要 2 个差异代谢物才能计算相关性热图。"
            mat_vals = df_scope[hm_feats].astype(float)
            mat_vals.columns = display_names
            corr_df = mat_vals.corr(method='pearson').fillna(0)
            
            do_row_cluster = (len(hm_feats) >= 3)
            do_col_cluster = (len(hm_feats) >= 3)
            cbar_title = "Pearson r"
            fig_side = max(8.0, min(16.5, len(hm_feats) * 0.34 + 3.2))
            
            g = sns.clustermap(
                corr_df, cmap="RdBu_r", vmin=-1.0, vmax=1.0, center=0,
                row_colors=row_colors_series, col_colors=row_colors_series,
                figsize=(fig_side, fig_side),
                row_cluster=do_row_cluster, col_cluster=do_col_cluster,
                dendrogram_ratio=(0.11, 0.11),
                colors_ratio=0.022,
                annot=(show_cell_values and len(hm_feats) <= 25), fmt=".2f"
            )
            plt.setp(
                g.ax_heatmap.get_xticklabels(),
                rotation=45 if len(hm_feats) <= 22 else 60,
                ha="right", rotation_mode="anchor", fontsize=8.5
            )
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=8.5)

            if row_colors_series is not None:
                legend_handles = [mpatches.Patch(color=mode_lut[m], label=f"Mode: {m}") for m in unique_modes]

        elif hm_type == "group_mean":
            grp_means = df_scope.groupby(group_col, sort=False)[hm_feats].mean()
            grp_means = grp_means.loc[[g for g in ordered_grps if g in grp_means.index]]
            hm_raw = grp_means.T.astype(float)
            hm_raw.index = display_names
            
            row_mean = hm_raw.mean(axis=1)
            row_std = hm_raw.std(axis=1).replace(0, 1.0)
            hm_z = hm_raw.sub(row_mean, axis=0).div(row_std, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)

            grp_lut = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(ordered_grps)}
            col_colors_series = pd.Series([grp_lut[g] for g in hm_z.columns], index=hm_z.columns)

            fig_w = max(7.0, min(14.0, len(ordered_grps) * 1.15 + 3.8))
            fig_h = max(6.8, min(16.5, len(hm_feats) * 0.32 + 2.8))
            g = sns.clustermap(
                hm_z, z_score=None, cmap="vlag", center=0,
                col_colors=col_colors_series, row_colors=row_colors_series,
                figsize=(fig_w, fig_h),
                row_cluster=do_row_cluster, col_cluster=False,
                dendrogram_ratio=(0.12, 0.01),
                colors_ratio=0.025,
                annot=show_cell_values, fmt=".2f"
            )
            plt.setp(g.ax_heatmap.get_xticklabels(), rotation=0, ha="center", fontsize=10, fontweight='bold')
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)

            legend_handles = [mpatches.Patch(color=grp_lut[grp], label=f"Group: {grp}") for grp in ordered_grps]
            if row_colors_series is not None:
                legend_handles += [mpatches.Patch(color=mode_lut[m], label=f"Mode: {m}") for m in unique_modes]

        else:
            sample_col_labels = make_unique(df_scope['SampleID'].astype(str).tolist())
            hm_raw = df_scope[hm_feats].T.astype(float)
            hm_raw.columns = sample_col_labels
            hm_raw.index = display_names
            
            row_mean = hm_raw.mean(axis=1)
            row_std = hm_raw.std(axis=1).replace(0, 1.0)
            hm_z = hm_raw.sub(row_mean, axis=0).div(row_std, axis=0).replace([np.inf, -np.inf], np.nan).fillna(0)
            
            grp_lut = {g: GROUP_COLORS[i % len(GROUP_COLORS)] for i, g in enumerate(ordered_grps)}
            col_colors_series = pd.Series(
                [grp_lut[str(g)] for g in df_scope[group_col]],
                index=sample_col_labels
            )
            
            n_samps = len(sample_col_labels)
            do_col_cluster = (hm_type == "sample_clustered") and (n_samps >= 3)
            fig_w = max(8.5, min(19.0, n_samps * 0.34 + 4.2))
            fig_h = max(6.8, min(16.5, len(hm_feats) * 0.32 + 2.8))
            
            g = sns.clustermap(
                hm_z, z_score=None, cmap="vlag", center=0, 
                col_colors=col_colors_series, row_colors=row_colors_series,
                figsize=(fig_w, fig_h),
                row_cluster=do_row_cluster, col_cluster=do_col_cluster,
                dendrogram_ratio=(0.11, 0.10 if do_col_cluster else 0.01),
                colors_ratio=0.024,
                annot=(show_cell_values and n_samps <= 15 and len(hm_feats) <= 25), fmt=".1f"
            )
            
            rot_angle = 45 if n_samps <= 18 else (60 if n_samps <= 36 else 90)
            ha_mode = "right" if rot_angle < 90 else "center"
            plt.setp(
                g.ax_heatmap.get_xticklabels(),
                rotation=rot_angle, ha=ha_mode,
                rotation_mode="anchor" if rot_angle < 90 else None,
                fontsize=8.5 if n_samps <= 28 else 7.5
            )
            plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9)

            legend_handles = [mpatches.Patch(color=grp_lut[grp], label=f"Group: {grp}") for grp in ordered_grps]
            if row_colors_series is not None:
                legend_handles += [mpatches.Patch(color=mode_lut[m], label=f"Mode: {m}") for m in unique_modes]

        g.ax_heatmap.set_xlabel("")
        g.ax_heatmap.set_ylabel("")
        if getattr(g, 'ax_row_colors', None) is not None:
            g.ax_row_colors.set_xticks([])
            g.ax_row_colors.set_xlabel("")
        if getattr(g, 'ax_col_colors', None) is not None:
            g.ax_col_colors.set_yticks([])
            g.ax_col_colors.set_ylabel("")
        if not do_col_cluster and getattr(g, 'ax_col_dendrogram', None) is not None:
            g.ax_col_dendrogram.set_visible(False)

        g.fig.canvas.draw()
        hm_box = g.ax_heatmap.get_position()
        left_edge = g.ax_row_dendrogram.get_position().x0 if (do_row_cluster and getattr(g, 'ax_row_dendrogram', None) is not None) else hm_box.x0
        
        top_edge = hm_box.y1
        if getattr(g, 'ax_col_colors', None) is not None:
            top_edge = max(top_edge, g.ax_col_colors.get_position().y1)
        if do_col_cluster and getattr(g, 'ax_col_dendrogram', None) is not None:
            top_edge = max(top_edge, g.ax_col_dendrogram.get_position().y1)

        if getattr(g, 'ax_cbar', None) is not None:
            cbar_w = 0.016
            cbar_h = min(0.26, hm_box.height * 0.42)
            cbar_x = left_edge - 0.055
            cbar_y = hm_box.y0 + (hm_box.height - cbar_h) * 0.5
            g.ax_cbar.set_position([cbar_x, cbar_y, cbar_w, cbar_h])
            g.ax_cbar.set_ylabel(cbar_title, fontsize=9.5, fontweight='bold', labelpad=6)
            g.ax_cbar.yaxis.set_ticks_position('left')
            g.ax_cbar.yaxis.set_label_position('left')
            g.ax_cbar.tick_params(labelsize=8.5)

        if legend_handles:
            ncol_leg = min(5, max(2, len(legend_handles)))
            g.fig.legend(
                handles=legend_handles,
                loc='lower left',
                bbox_to_anchor=(hm_box.x0, top_edge + 0.02),
                ncol=ncol_leg,
                frameon=True,
                facecolor='white',
                edgecolor='#cccccc',
                fontsize=9
            )

        buf_png = io.BytesIO()
        g.savefig(buf_png, format='png', dpi=300, bbox_inches='tight', pad_inches=0.25)
        png_bytes = buf_png.getvalue()

        buf_pdf = io.BytesIO()
        g.savefig(buf_pdf, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0.25)
        pdf_bytes = buf_pdf.getvalue()

        plt.close(g.fig)
        return png_bytes, pdf_bytes, None
    except Exception as e:
        plt.close('all')
        return None, None, f"热图绘制异常: {str(e)}"


# ==========================================
# 辅助函数 3：单矩阵统计建模与图表生成 (对齐 MetaboAnalyst Log2_FC 与 T 检验)
# ==========================================
def run_single_matrix_modeling(
    df_proc, df_unscaled_log2, feats, meta, group_col,
    sel_grps, case, ctrl, p_th, fc_th, vip_th, vip_show_num,
    equal_var=True, mode_title=""
):
    prefix = f"[{mode_title}] " if mode_title else ""
    df_sub = df_proc[df_proc[group_col].isin(sel_grps)].copy()
    df_log_sub = df_unscaled_log2[df_unscaled_log2[group_col].isin(sel_grps)].copy()
    
    linear_full = df_unscaled_log2.attrs.get('linear_norm_df', None)
    df_lin_sub = linear_full[linear_full[group_col].isin(sel_grps)].copy() if linear_full is not None else None
    
    if len(df_sub) < 4: raise ValueError(f"{prefix}样本数太少 ({len(df_sub)}个)，无法建模。")
    if len(feats) < 2: raise ValueError(f"{prefix}有效特征太少 ({len(feats)}个)，无法建模。")

    stats_df = run_pairwise_statistics(
        df_sub, group_col, case, ctrl, feats,
        equal_var=equal_var, df_unscaled_log2=df_log_sub, df_linear_norm=df_lin_sub
    )
    
    if meta is not None and 'Clean_Name' in meta.columns: 
        merge_cols = [c for c in ['Clean_Name', 'Original_Name', 'KEGG_ID', 'Source_Mode', 'Detected_Modes'] if c in meta.columns]
        meta_dedup = meta[~meta.index.duplicated(keep='first')]
        stats_df = stats_df.merge(meta_dedup[merge_cols], left_on='Metabolite', right_index=True, how='left')
        stats_df['Name'] = stats_df['Clean_Name'].fillna(stats_df['Metabolite'])
        if 'Source_Mode' not in stats_df.columns:
            stats_df['Source_Mode'] = mode_title or 'Global'
        else:
            stats_df['Source_Mode'] = stats_df['Source_Mode'].fillna(mode_title or 'Global')
        if 'Detected_Modes' not in stats_df.columns:
            stats_df['Detected_Modes'] = stats_df['Source_Mode']
        else:
            stats_df['Detected_Modes'] = stats_df['Detected_Modes'].fillna(stats_df['Source_Mode'])
        
        if 'KEGG_ID' in stats_df.columns:
            kegg_str = stats_df['KEGG_ID'].fillna('').astype(str).replace('nan', '').replace('None', '')
            orig_str = stats_df['Original_Name'].fillna(stats_df['Name']).astype(str)
            stats_df['Search_Name'] = np.where(kegg_str != '', orig_str + '|' + kegg_str, orig_str)
        else:
            stats_df['Search_Name'] = stats_df['Original_Name'].fillna(stats_df['Name'])
    else:
        stats_df['Name'] = stats_df['Search_Name'] = stats_df['Metabolite']
        stats_df['Source_Mode'] = mode_title or 'Global'
        stats_df['Detected_Modes'] = stats_df['Source_Mode']

    stats_df['Sig'] = 'NS'
    stats_df.loc[(stats_df['P_Value'] < p_th) & (stats_df['Log2_FC'] > fc_th), 'Sig'] = 'Up'
    stats_df.loc[(stats_df['P_Value'] < p_th) & (stats_df['Log2_FC'] < -fc_th), 'Sig'] = 'Down'

    y_binary = np.where(df_sub[group_col] == case, 1, -1)
    X_matrix = df_sub[feats].values
    
    opls = OPLS_DA()
    opls.fit(X_matrix, y_binary)
    corrs, r2_perm, q2_perm, R2Y, Q2 = opls.permutation_test(X_matrix, y_binary, n_permutations=100)
    
    m_q2, b_q2 = np.polyfit(corrs, q2_perm, 1) if len(corrs) > 0 else (0, 0)
    m_r2, b_r2 = np.polyfit(corrs, r2_perm, 1) if len(corrs) > 0 else (0, 0)

    vip_df = pd.DataFrame({'Metabolite': feats, 'VIP': opls.vip, 'p_corr': opls.p_corr})
    stats_df = stats_df.merge(vip_df, on='Metabolite')
    stats_df['Name_With_Mode'] = stats_df['Name'].astype(str) + " [" + stats_df['Source_Mode'].astype(str) + "]"
    stats_df['Is_Biomarker'] = (stats_df['VIP'] >= vip_th) & (stats_df['P_Value'] < p_th) & (stats_df['Log2_FC'].abs() > fc_th)
    out_df = stats_df[stats_df['Is_Biomarker']].sort_values('VIP', ascending=False).copy()

    # 1. OPLS-DA 得分图
    opls_score_df = pd.DataFrame({'t1 (Predictive)': opls.t, 't_ortho (Orthogonal)': opls.t_ortho, 'Group': df_sub[group_col].values, 'SampleID': df_sub['SampleID'].values})
    fig_opls = px.scatter(opls_score_df, x='t1 (Predictive)', y='t_ortho (Orthogonal)', color='Group', symbol='Group', hover_data=['SampleID'], color_discrete_sequence=GROUP_COLORS)
    for i, g in enumerate(list(sel_grps)):
        sub_grp = opls_score_df[opls_score_df['Group'] == g]
        if len(sub_grp) >= 3:
            el_x, el_y = get_ellipse_coordinates(sub_grp['t1 (Predictive)'], sub_grp['t_ortho (Orthogonal)'])
            if el_x is not None: fig_opls.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color=GROUP_COLORS[i % len(GROUP_COLORS)], width=2, dash='dash'), showlegend=False, hoverinfo='skip'))
    fig_opls.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
    fig_opls.add_hline(y=0, line_dash="dash", line_color="gray"); fig_opls.add_vline(x=0, line_dash="dash", line_color="gray")
    fig_opls = update_layout_square(fig_opls, f"{prefix}OPLS-DA Score Plot", "t [1]", "to [1]")

    # 2. 置换检验图
    fig_perm = go.Figure()
    fig_perm.add_trace(go.Scatter(x=corrs, y=r2_perm, mode='markers', name='R2', marker=dict(color='green', symbol='circle-open', size=8)))
    fig_perm.add_trace(go.Scatter(x=corrs, y=q2_perm, mode='markers', name='Q2', marker=dict(color='blue', symbol='square-open', size=8)))
    fig_perm.add_trace(go.Scatter(x=[1], y=[R2Y], mode='markers', name='Original R2', marker=dict(color='green', symbol='circle', size=12)))
    fig_perm.add_trace(go.Scatter(x=[1], y=[Q2], mode='markers', name='Original Q2', marker=dict(color='blue', symbol='square', size=12)))
    x_line = np.array([0, 1])
    fig_perm.add_trace(go.Scatter(x=x_line, y=m_r2*x_line + b_r2, mode='lines', name=f'R2 Line (Int: {b_r2:.2f})', line=dict(color='green', dash='dash')))
    fig_perm.add_trace(go.Scatter(x=x_line, y=m_q2*x_line + b_q2, mode='lines', name=f'Q2 Line (Int: {b_q2:.2f})', line=dict(color='blue', dash='dash')))
    fig_perm.update_layout(template="simple_white", width=600, height=600, title={'text': f"{prefix}Permutation Test", 'y':0.95, 'x':0.5, 'xanchor': 'center'}, xaxis_title="Correlation", yaxis_title="R2 / Q2")

    # 3. S-Plot
    splot_df = stats_df.copy()
    sig_label_str = f'VIP≥{vip_th} & P<{p_th}'
    splot_df['Color'] = np.where(splot_df['Is_Biomarker'], sig_label_str, 'NS')
    fig_splot = px.scatter(
        splot_df, x='Log2_FC', y='p_corr', color='Color',
        hover_name='Name_With_Mode', hover_data=['Source_Mode', 'VIP', 'P_Value'],
        color_discrete_map={sig_label_str: '#CD0000', 'NS': '#E0E0E0'}
    )
    fig_splot.add_hline(y=0.5, line_dash="dash", line_color="gray"); fig_splot.add_hline(y=-0.5, line_dash="dash", line_color="gray")
    fig_splot = update_layout_square(fig_splot, f"{prefix}S-Plot", "Log2 Fold Change", "p(corr)")

    # 4. VIP 图
    top_vip_df = stats_df.sort_values('VIP', ascending=True).tail(vip_show_num).copy()
    top_vip_df['VIP_Label'] = make_unique(top_vip_df['Name_With_Mode'].tolist())
    fig_vip = px.bar(top_vip_df, x="VIP", y="VIP_Label", orientation='h', color="VIP", hover_data=['Source_Mode', 'Log2_FC', 'P_Value'], color_continuous_scale="RdBu_r")
    if vip_th > 0:
        fig_vip.add_vline(x=vip_th, line_dash="dash", line_color="black")
    fig_vip.update_layout(template="simple_white", width=820, height=700, title={'text': f"{prefix}Top VIP Scores", 'x':0.5, 'xanchor': 'center'}, yaxis_title="Metabolite [Mode]", coloraxis_showscale=False)

    # 5. PCA 图
    fig_pca = None
    if len(df_proc) >= 3:
        valid_feats_pca = df_proc[feats].var()[df_proc[feats].var() > 1e-9].index.tolist()
        if len(valid_feats_pca) >= 2:
            X_pca_input = df_proc[valid_feats_pca].values
            pca_all = PCA(n_components=2).fit(X_pca_input)
            pcs_all = pca_all.transform(X_pca_input)
            var_all = pca_all.explained_variance_ratio_
            pca_df_all = pd.DataFrame({'PC1': pcs_all[:, 0], 'PC2': pcs_all[:, 1], 'Group': df_proc[group_col].values, 'SampleID': df_proc['SampleID']})
            fig_pca = px.scatter(pca_df_all, x='PC1', y='PC2', color='Group', symbol='Group', hover_data=['SampleID'], color_discrete_sequence=GROUP_COLORS)
            for i, g in enumerate(sorted(df_proc[group_col].unique())):
                sub_grp = pca_df_all[pca_df_all['Group'] == g]
                if len(sub_grp) >= 3:
                    el_x, el_y = get_ellipse_coordinates(sub_grp['PC1'], sub_grp['PC2'])
                    if el_x is not None: fig_pca.add_trace(go.Scatter(x=el_x, y=el_y, mode='lines', line=dict(color=GROUP_COLORS[i % len(GROUP_COLORS)], width=1, dash='dot'), showlegend=False, hoverinfo='skip'))
            fig_pca.update_traces(marker=dict(size=14, line=dict(width=1, color='black'), opacity=0.9))
            fig_pca.add_hline(y=0, line_dash="dash", line_color="gray"); fig_pca.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_pca = update_layout_square(fig_pca, f"{prefix}PCA Plot", f"PC1 ({var_all[0]:.1%})", f"PC2 ({var_all[1]:.1%})")

    # 6. 默认火山图
    fig_vol = build_volcano_figure(stats_df, p_th, fc_th, title=f"{prefix}Volcano Plot", show_labels=False, top_label_n=10)

    return {
        'df_sub': df_sub, 'stats_df': stats_df, 'out_df': out_df,
        'R2Y': R2Y, 'Q2': Q2, 'b_q2': b_q2,
        'fig_opls': fig_opls, 'fig_perm': fig_perm, 'fig_splot': fig_splot,
        'fig_vip': fig_vip, 'fig_pca': fig_pca, 'fig_vol': fig_vol
    }


# ==========================================
# 4. 执行核心分析计算 (支持模式 A 与模式 B 双轨运行)
# ==========================================
if submit_button:
    if len(sel_grps) != 2 or case == ctrl: 
        st.error("⚠️ 必须选择 2 个不同的对比组！")
        st.stop()

    plt.close('all')
    use_mode_b = integration_mode.startswith("模式 B") and (len(mode_datasets) > 1)
    bg_mode_key = "reference" if kegg_bg_option.startswith("物种全库") else "experimental"
    filter_mode_code = 'per_group' if miss_filter_rule.startswith("组内修正") else 'global'
    equal_var_flag = ttest_var_label.startswith("Student")
    impute_code_map = {
        "1/5 Min (LoD, 同 MetaboAnalyst 默认)": "lod_1_5",
        "1/2 Min": "min",
        "KNN (K近邻插补)": "KNN",
        "mean (均值)": "mean",
        "zero (极小常数)": "zero"
    }
    impute_m = impute_code_map.get(impute_label, "lod_1_5")

    with st.spinner("正在运行代谢组学统计与通路分析引擎..."):
        try:
            mode_results = {}
            mode_summary_rows = []
            combined_all_stats = None

            if not use_mode_b:
                # ==========================================================
                # 轨道 A：各模式独立归一化 ➔ 跨模式优选去重 ➔ 全局统一建模
                # ==========================================================
                if len(mode_datasets) > 1:
                    normed_mode_list = []
                    for m_df, m_meta, m_tag in mode_datasets:
                        m_norm, m_feats = data_cleaning_pipeline(
                            m_df, group_col, missing_thresh=miss_th,
                            impute_method=impute_m, norm_method=norm_m,
                            log_transform=False, scale_method='None',
                            target_groups=sel_grps, filter_mode=filter_mode_code
                        )
                        m_meta_sub = m_meta.loc[m_meta.index.intersection(m_feats)].copy()
                        normed_mode_list.append((m_norm, m_meta_sub, m_tag))
                    
                    merged_norm_df, meta, err = merge_multiple_dfs(normed_mode_list)
                    if err: raise RuntimeError(err)
                    grp_map = dict(zip(st.session_state.raw_df['SampleID'], st.session_state.raw_df[group_col]))
                    merged_norm_df[group_col] = merged_norm_df['SampleID'].map(grp_map).fillna('Unknown')
                    df_proc, feats, df_unscaled_log2 = data_cleaning_pipeline(
                        merged_norm_df, group_col, missing_thresh=1.0,
                        impute_method=impute_m, norm_method='None',
                        log_transform=do_log, scale_method=scale_m,
                        return_unscaled=True, target_groups=sel_grps, filter_mode=filter_mode_code
                    )
                else:
                    raw_df = st.session_state.raw_df
                    meta = st.session_state.feature_meta
                    df_proc, feats, df_unscaled_log2 = data_cleaning_pipeline(
                        raw_df, group_col, missing_thresh=miss_th, 
                        impute_method=impute_m, norm_method=norm_m, 
                        log_transform=do_log, scale_method=scale_m,
                        return_unscaled=True, target_groups=sel_grps, filter_mode=filter_mode_code
                    )

                global_res = run_single_matrix_modeling(
                    df_proc, df_unscaled_log2, feats, meta, group_col,
                    sel_grps, case, ctrl, p_th, fc_th, vip_th, vip_show_num,
                    equal_var=equal_var_flag
                )
                df_sub = global_res['df_sub']
                df_all_unscaled = df_unscaled_log2.copy()
                stats_df = global_res['stats_df']
                out_df = global_res['out_df']
                R2Y, Q2, b_q2 = global_res['R2Y'], global_res['Q2'], global_res['b_q2']
                fig_opls, fig_perm = global_res['fig_opls'], global_res['fig_perm']
                fig_splot, fig_vip = global_res['fig_splot'], global_res['fig_vip']
                fig_pca, fig_vol = global_res['fig_pca'], global_res['fig_vol']
                bg_universe_list = stats_df['Search_Name'].unique().tolist()
                integration_label = "模式 A：单模式独立归一化 ➔ 跨模式优选去重 ➔ 全局统一建模"

            else:
                # ==========================================================
                # 轨道 B：各模式独立统计建模 ➔ 差异物按最小 P 值裁决去重 ➔ 通路富集
                # ==========================================================
                all_mode_out_dfs = []
                all_mode_stats_dfs = []
                all_mode_bg_names = set()
                scaled_sub_dfs = []
                unscaled_all_dfs = []

                for m_df, m_meta, m_tag in mode_datasets:
                    n_raw_m = len([c for c in m_df.columns if c not in ['SampleID', 'Group', 'Source_Files']])
                    m_proc, m_feats, m_unscaled_log2 = data_cleaning_pipeline(
                        m_df, group_col, missing_thresh=miss_th,
                        impute_method=impute_m, norm_method=norm_m,
                        log_transform=do_log, scale_method=scale_m,
                        return_unscaled=True, target_groups=sel_grps, filter_mode=filter_mode_code
                    )
                    m_res = run_single_matrix_modeling(
                        m_proc, m_unscaled_log2, m_feats, m_meta, group_col,
                        sel_grps, case, ctrl, p_th, fc_th, vip_th, vip_show_num,
                        equal_var=equal_var_flag, mode_title=m_tag
                    )
                    mode_results[m_tag] = m_res
                    
                    n_up = int((m_res['out_df']['Log2_FC'] > 0).sum()) if not m_res['out_df'].empty else 0
                    n_down = int((m_res['out_df']['Log2_FC'] < 0).sum()) if not m_res['out_df'].empty else 0
                    diag_str = "优秀" if (m_res['b_q2'] < 0.05 and m_res['Q2'] > 0.5) else ("局部差异(未过拟合)" if m_res['b_q2'] < 0.05 else "过拟合风险")
                    mode_summary_rows.append({
                        '采集模式': m_tag,
                        '原始特征数': n_raw_m,
                        '清洗保留特征数': len(m_feats),
                        '显著差异物总数': len(m_res['out_df']),
                        '上调 (Up)': n_up,
                        '下调 (Down)': n_down,
                        'R2Y': m_res['R2Y'],
                        'Q2': m_res['Q2'],
                        'Q2截距': m_res['b_q2'],
                        '模型诊断': diag_str
                    })
                    all_mode_bg_names.update(m_res['stats_df']['Search_Name'].dropna().unique().tolist())
                    all_mode_stats_dfs.append(m_res['stats_df'])
                    if not m_res['out_df'].empty:
                        all_mode_out_dfs.append(m_res['out_df'])
                    
                    m_sub_idx = m_res['df_sub'].drop_duplicates(subset=['SampleID']).set_index('SampleID')
                    scaled_sub_dfs.append((m_sub_idx, m_feats))
                    
                    m_all_idx = m_unscaled_log2.drop_duplicates(subset=['SampleID']).set_index('SampleID')
                    unscaled_all_dfs.append((m_all_idx, m_feats))

                if all_mode_stats_dfs:
                    combined_all_stats = pd.concat(all_mode_stats_dfs, ignore_index=True)

                if all_mode_out_dfs:
                    combined_sig = pd.concat(all_mode_out_dfs, ignore_index=True)
                    sig_modes_map = combined_sig.groupby('Name')['Source_Mode'].apply(lambda s: ", ".join(list(dict.fromkeys(s.astype(str))))).to_dict()
                    combined_sig = combined_sig.sort_values(['P_Value', 'VIP'], ascending=[True, False])
                    out_df = combined_sig.drop_duplicates(subset=['Name'], keep='first').copy()
                    out_df['Detected_Modes'] = out_df['Name'].map(sig_modes_map)
                    out_df['Name_With_Mode'] = out_df['Name'].astype(str) + " [" + out_df['Source_Mode'].astype(str) + "]"
                    out_df = out_df.sort_values('VIP', ascending=False).reset_index(drop=True)
                else:
                    out_df = pd.DataFrame(columns=['Metabolite', 'Name', 'Name_With_Mode', 'Search_Name', 'Source_Mode', 'Detected_Modes', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr'])

                base_sub = mode_results[list(mode_results.keys())[0]]['df_sub'][['SampleID', group_col]].drop_duplicates(subset=['SampleID']).set_index('SampleID')
                feat_frames = [s_df[s_feats] for s_df, s_feats in scaled_sub_dfs]
                df_sub = pd.concat([base_sub] + feat_frames, axis=1, join='outer').fillna(0).reset_index()

                base_all = st.session_state.raw_df[['SampleID', group_col]].drop_duplicates(subset=['SampleID']).set_index('SampleID')
                all_feat_frames = [u_df[u_feats] for u_df, u_feats in unscaled_all_dfs]
                df_all_unscaled = pd.concat([base_all] + all_feat_frames, axis=1, join='outer').fillna(0).reset_index()
                stats_df = combined_all_stats if combined_all_stats is not None else pd.DataFrame()

                best_m = max(mode_results.keys(), key=lambda k: mode_results[k]['Q2'])
                R2Y, Q2, b_q2 = mode_results[best_m]['R2Y'], mode_results[best_m]['Q2'], mode_results[best_m]['b_q2']
                fig_opls, fig_perm = mode_results[best_m]['fig_opls'], mode_results[best_m]['fig_perm']
                fig_splot, fig_pca = mode_results[best_m]['fig_splot'], mode_results[best_m]['fig_pca']
                fig_vol = build_volcano_figure(combined_all_stats, p_th, fc_th, title="Combined Volcano Plot (All Modes)", show_labels=False) if combined_all_stats is not None else mode_results[best_m]['fig_vol']

                if not out_df.empty:
                    top_vip_merged = out_df.sort_values('VIP', ascending=True).tail(vip_show_num).copy()
                    top_vip_merged['Display_Name'] = make_unique(top_vip_merged['Name_With_Mode'].tolist())
                    fig_vip = px.bar(top_vip_merged, x="VIP", y="Display_Name", orientation='h', color="VIP", hover_data=['Detected_Modes', 'Log2_FC', 'P_Value'], color_continuous_scale="RdBu_r")
                    if vip_th > 0:
                        fig_vip.add_vline(x=vip_th, line_dash="dash", line_color="black")
                    fig_vip.update_layout(template="simple_white", width=820, height=700, title={'text': "Merged Top Biomarkers VIP (Across Modes)", 'x':0.5, 'xanchor': 'center'}, yaxis_title="Metabolite [Best Mode]", coloraxis_showscale=False)
                else:
                    fig_vip = mode_results[best_m]['fig_vip']

                bg_universe_list = list(all_mode_bg_names)
                integration_label = "模式 B：各采集模式独立统计建模 ➔ 差异代谢物去重合并 (JHM 经典流派)"

            mode_summary_df = pd.DataFrame(mode_summary_rows) if mode_summary_rows else None

            # ==========================================================
            # 公共下游 1：生成默认热图供离线 HTML 报告嵌入
            # ==========================================================
            hm_base64 = ""
            png_bytes_init, _, _ = generate_custom_heatmap(
                df_all_unscaled, out_df, group_col, sel_grps,
                scope_mode="pairwise", hm_type="sample_sorted", top_n=40, show_row_mode=True
            )
            if png_bytes_init:
                hm_base64 = base64.b64encode(png_bytes_init).decode('utf-8')

            # ==========================================================
            # 公共下游 2：诊断列线图 (Nomogram)
            # ==========================================================
            fig_nomogram = None
            sig_mets = [m for m in out_df['Metabolite'].tolist() if m in df_sub.columns]
            if len(sig_mets) >= 2:
                try:
                    nomo_sub_df = out_df[out_df['Metabolite'].isin(sig_mets)].head(min(nomo_num, len(sig_mets)))
                    nomo_labels = nomo_sub_df.get('Name_With_Mode', nomo_sub_df['Name']).tolist()
                    fig_nomogram = plot_nomogram(df_sub, nomo_sub_df['Metabolite'].tolist(), nomo_labels, group_col, case)
                except Exception:
                    pass

            # ==========================================================
            # 公共下游 3：KEGG 通路富集与机制网络图
            # ==========================================================
            pathway_df, filtered_db_df, fig_pathway, fig_network = pd.DataFrame(), pd.DataFrame(), None, None
            sig_mets_fullnames = out_df['Search_Name'].dropna().unique().tolist()
            if sig_mets_fullnames:
                pathway_df, filtered_db_df = run_pathway_enrichment(
                    sig_mets_fullnames, bg_universe_list,
                    custom_db_source=custom_pathway_file if custom_pathway_file else db_filename,
                    bg_mode=bg_mode_key
                )
                if not pathway_df.empty and 'Error' not in pathway_df.columns:
                    plot_pw_df = pathway_df[pathway_df['Hits'] > 0].head(pw_show_num)
                    fig_pathway = px.scatter(
                        plot_pw_df, x='Enrichment_Factor', y='-Log10_P', size='Hits', color='P_Value',
                        hover_name='Pathway', hover_data={'Hit_Metabolites': True, 'P_Value': ':.4f', 'FDR': ':.4f'},
                        color_continuous_scale='Reds_r', size_max=40
                    )
                    fig_pathway.update_traces(marker=dict(line=dict(width=1, color='black'), opacity=0.75))
                    fig_pathway.add_hline(y=-np.log10(0.05), line_dash="dash", line_color="gray")
                    fig_pathway.update_layout(template="simple_white", width=800, height=600, title={'text': "KEGG Pathway Enrichment", 'x':0.5, 'xanchor': 'center'}, xaxis_title="Enrichment Factor", yaxis_title="-Log10(P)")
                    
                    sig_pws = pathway_df[pathway_df['P_Value'] < 0.05].head(pw_show_num)
                    if sig_pws.empty: sig_pws = pathway_df.head(min(5, len(pathway_df)))
                    if not sig_pws.empty:
                        G = nx.Graph()
                        fc_dict = {}
                        mode_dict = {}
                        for _, r_row in out_df.iterrows():
                            k1 = str(r_row['Name']).split(';')[0].strip()
                            k2 = str(r_row['Search_Name']).split('|')[0].split(';')[0].strip()
                            fc_dict[k1] = r_row['Log2_FC']; fc_dict[k2] = r_row['Log2_FC']
                            mode_dict[k1] = r_row.get('Source_Mode', ''); mode_dict[k2] = r_row.get('Source_Mode', '')
                            
                        for _, row in sig_pws.iterrows():
                            pw_name = row['Pathway']
                            G.add_node(pw_name, node_type='pathway', size=max(16, min(38, -np.log10(row['P_Value']) * 10)), disp_name=pw_name)
                            if pd.notna(row['Hit_Metabolites']) and str(row['Hit_Metabolites']).strip() != "":
                                for hit in [m.strip() for m in str(row['Hit_Metabolites']).split(',') if m.strip()]:
                                    fc_val = fc_dict.get(hit, 0.0)
                                    m_tag_str = mode_dict.get(hit, '')
                                    disp_label = f"{hit} [{m_tag_str}]" if m_tag_str else hit
                                    G.add_node(hit, node_type='metabolite', size=11, fc=fc_val, disp_name=disp_label)
                                    G.add_edge(pw_name, hit)
                                    
                        if len(G.nodes) > 0:
                            pos = nx.spring_layout(G, k=0.75, seed=42)
                            edge_x, edge_y = [], []
                            for e0, e1 in G.edges():
                                edge_x.extend([pos[e0][0], pos[e1][0], None])
                                edge_y.extend([pos[e0][1], pos[e1][1], None])
                                
                            edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=1.2, color='#999'), hoverinfo='none')
                            node_trace = go.Scatter(
                                x=[pos[n][0] for n in G.nodes()],
                                y=[pos[n][1] for n in G.nodes()],
                                mode='markers+text',
                                text=[G.nodes[n].get('disp_name', n) for n in G.nodes()],
                                textposition="top center",
                                marker=dict(
                                    symbol=['square' if G.nodes[n]['node_type']=='pathway' else 'circle' for n in G.nodes()],
                                    color=['#FFD700' if G.nodes[n]['node_type']=='pathway' else ('#CD0000' if G.nodes[n]['fc'] >= 0 else '#00008B') for n in G.nodes()],
                                    size=[G.nodes[n]['size'] for n in G.nodes()],
                                    line_width=1, line_color='black'
                                )
                            )
                            fig_network = go.Figure(data=[edge_trace, node_trace])
                            fig_network.update_layout(title={'text': "Mechanism Network (Pathway - Metabolite [Mode])", 'y':0.95, 'x':0.5, 'xanchor': 'center'}, showlegend=False, xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), width=900, height=700, plot_bgcolor='white')

            # 为 report_generator 兼容保留 '特征数' 别名列
            if mode_summary_df is not None and '清洗保留特征数' in mode_summary_df.columns:
                mode_summary_df['特征数'] = mode_summary_df['清洗保留特征数']

            html_report = generate_offline_html(
                case, ctrl, feats if not use_mode_b else [], p_th, fc_th, norm_m, scale_m, R2Y, Q2, b_q2,
                out_df, pathway_df, fig_opls, fig_perm, fig_splot, fig_vip, fig_vol, fig_pca,
                hm_base64, fig_nomogram, fig_pathway, fig_network, vip_show_num, pw_show_num, nomo_num,
                mode_results=mode_results if use_mode_b else None,
                mode_summary_df=mode_summary_df,
                integration_label=integration_label
            )
            prompt_md = generate_ai_prompt(case, ctrl, norm_m, scale_m, R2Y, Q2, b_q2, p_th, fc_th, out_df, pathway_df, mode_summary_df=mode_summary_df)

            st.session_state['analysis_res'] = {
                'case': case, 'ctrl': ctrl, 'sel_grps': sel_grps, 'group_col': group_col,
                'p_th': p_th, 'fc_th': fc_th, 'vip_th': vip_th,
                'retained_count': len(stats_df),
                'R2Y': R2Y, 'Q2': Q2, 'b_q2': b_q2,
                'use_mode_b': use_mode_b, 'mode_results': mode_results, 'mode_summary_df': mode_summary_df,
                'stats_df': stats_df, 'combined_all_stats': combined_all_stats,
                'df_all_unscaled': df_all_unscaled,
                'fig_opls': fig_opls, 'fig_perm': fig_perm, 'fig_splot': fig_splot,
                'fig_vip': fig_vip, 'fig_pca': fig_pca, 'fig_vol': fig_vol,
                'hm_base64': hm_base64, 'out_df': out_df, 'pathway_df': pathway_df,
                'filtered_db_df': filtered_db_df, 
                'fig_nomogram': fig_nomogram, 'fig_pathway': fig_pathway, 'fig_network': fig_network,
                'html_report': html_report, 'prompt_md': prompt_md
            }
        except Exception as e:
            st.error(f"❌ 分析计算失败。错误摘要: {str(e)}")
            with st.expander("点击查看详细报错日志"): st.code(traceback.format_exc())

# ==========================================
# 5. UI 展示层 (含火山图标签切换开关 & 零重叠全样本热图工作台)
# ==========================================
if 'analysis_res' in st.session_state:
    res = st.session_state['analysis_res']
    safe_case_str = safe_filename_part(res['case'])
    safe_ctrl_str = safe_filename_part(res['ctrl'])
    
    st.title("📊 综合代谢组学分析报告")
    if res.get('use_mode_b') and res.get('mode_summary_df') is not None:
        st.markdown(f"**对比**: `{res['case']}` vs `{res['ctrl']}` &nbsp;&nbsp;|&nbsp;&nbsp; **整合流派**: `模式 B (各采集模式独立统计建模 ➔ 差异物去重富集)` &nbsp;&nbsp;|&nbsp;&nbsp; **去重后总差异代谢物**: `{len(res['out_df'])}` 个")
        disp_sum_df = res['mode_summary_df'].drop(columns=['特征数'], errors='ignore')
        st.dataframe(disp_sum_df.style.format({"R2Y": "{:.3f}", "Q2": "{:.3f}", "Q2截距": "{:.3f}"}), width='stretch')
    else:
        st.markdown(
            f"**对比**: `{res['case']}` vs `{res['ctrl']}` &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**清洗后保留特征**: `{res.get('retained_count', len(res['stats_df']))}` 个 &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**模型**: R²Y = `{res['R2Y']:.3f}` &nbsp;|&nbsp; Q² = `{res['Q2']:.3f}` &nbsp;&nbsp;|&nbsp;&nbsp; "
            f"**显著差异物**: `{len(res['out_df'])}` 个"
        )
        b_q2_val, q2_val = res['b_q2'], res['Q2']
        if b_q2_val < 0.05 and q2_val > 0.5:
            st.success(f"✅ OPLS-DA 模型预测能力强，且未发生过拟合 (Q²={q2_val:.3f}, 截距={b_q2_val:.3f})")
        elif b_q2_val < 0.05 and q2_val <= 0.5:
            st.info(f"💡 模型未过拟合，但组间整体代谢差异偏弱 (Q²={q2_val:.3f} < 0.5, 截距={b_q2_val:.3f})")
        else:
            st.warning(f"⚠️ 警告：模型存在严重的过拟合风险，请谨慎解读 VIP 值！(Q²截距={b_q2_val:.3f} ≥ 0.05)")

    def render_mode_chart(fig_key):
        """若为模式 B 则自动生成各采集模式的切换子标签页"""
        if res.get('use_mode_b') and res.get('mode_results'):
            m_names = list(res['mode_results'].keys())
            sub_tabs = st.tabs([f"🔬 {m}" for m in m_names])
            for idx, m in enumerate(m_names):
                with sub_tabs[idx]:
                    fig_obj = res['mode_results'][m].get(fig_key)
                    if fig_obj: st.plotly_chart(fig_obj, key=f"{fig_key}_{m}")
                    else: st.warning("图表生成失败")
        else:
            if res.get(fig_key): st.plotly_chart(res[fig_key], key=f"{fig_key}_global")
            else: st.warning("图表生成失败")

    tabs = st.tabs([
        "🎯 OPLS-DA",
        "🔄 置换检验",
        "🧬 S-Plot",
        "📊 VIP",
        "🌐 PCA",
        "🌋 火山图",
        "🔥 聚类热图",
        "📑 清单",
        "📏 列线图",
        "🕸️ 通路富集",
        "🔗 机制网络图",
        "📄 导出报告"
    ])
    
    with tabs[0]:
        c1, c2 = st.columns([1, 5])
        with c2: render_mode_chart('fig_opls')
            
    with tabs[1]:
        c1, c2 = st.columns([1, 5])
        with c2: render_mode_chart('fig_perm')
            
    with tabs[2]:
        c1, c2 = st.columns([1, 5])
        with c2: render_mode_chart('fig_splot')
            
    with tabs[3]:
        c1, c2 = st.columns([1, 6])
        with c2: 
            if res.get('use_mode_b') and res.get('mode_results'):
                m_names = list(res['mode_results'].keys())
                vip_tabs = st.tabs(["🏆 跨模式合并 Top VIP"] + [f"🔬 {m}" for m in m_names])
                with vip_tabs[0]:
                    if res['fig_vip']: st.plotly_chart(res['fig_vip'], key="vip_main")
                for idx, m in enumerate(m_names):
                    with vip_tabs[idx + 1]:
                        if res['mode_results'][m].get('fig_vip'):
                            st.plotly_chart(res['mode_results'][m]['fig_vip'], key=f"vip_{m}")
            else:
                if res['fig_vip']: st.plotly_chart(res['fig_vip'], key="vip_main")
                else: st.warning("图表生成失败")
            
    with tabs[4]:
        c1, c2 = st.columns([1, 5])
        with c2: render_mode_chart('fig_pca')
            
    # ==========================================================
    # 独立标签页 1：🌋 火山图 (带化合物名称显隐切换按钮，默认关闭防遮挡)
    # ==========================================================
    with tabs[5]:
        v_col1, v_col2 = st.columns([2, 3])
        with v_col1:
            show_vol_labels = st.checkbox(
                "🏷️ 在火山图上直接显示化合物名称标签 (关闭时仅鼠标悬停显示，避免遮挡散点)",
                value=False,
                key="toggle_vol_labels"
            )
        with v_col2:
            vol_top_n = st.slider(
                "显示 Top 显著差异物标签数量",
                min_value=1, max_value=30, value=10,
                disabled=(not show_vol_labels),
                key="slider_vol_top_n"
            )
            
        c_v1, c_v2 = st.columns([1, 6])
        with c_v2:
            p_th_val = res.get('p_th', 0.05)
            fc_th_val = res.get('fc_th', 0.58)
            if res.get('use_mode_b') and res.get('mode_results'):
                m_names = list(res['mode_results'].keys())
                vol_sub_tabs = st.tabs(["🌐 四模式汇总火山图"] + [f"🔬 {m}" for m in m_names])
                with vol_sub_tabs[0]:
                    if res.get('combined_all_stats') is not None:
                        fig_v_merged = build_volcano_figure(
                            res['combined_all_stats'], p_th_val, fc_th_val,
                            title="Combined Volcano Plot (All Modes)",
                            show_labels=show_vol_labels, top_label_n=vol_top_n
                        )
                        st.plotly_chart(fig_v_merged, key="vol_dyn_merged")
                for idx, m in enumerate(m_names):
                    with vol_sub_tabs[idx + 1]:
                        m_st_df = res['mode_results'][m]['stats_df']
                        fig_v_m = build_volcano_figure(
                            m_st_df, p_th_val, fc_th_val,
                            title=f"[{m}] Volcano Plot",
                            show_labels=show_vol_labels, top_label_n=vol_top_n
                        )
                        st.plotly_chart(fig_v_m, key=f"vol_dyn_{m}")
            else:
                fig_v_global = build_volcano_figure(
                    res['stats_df'], p_th_val, fc_th_val,
                    title="Volcano Plot",
                    show_labels=show_vol_labels, top_label_n=vol_top_n
                )
                st.plotly_chart(fig_v_global, key="vol_dyn_global")
            
    # ==========================================================
    # 独立标签页 2：🔥 聚类热图工作台 (零重叠布局 + PNG/PDF双格式下载)
    # ==========================================================
    with tabs[6]:
        st.markdown("#### 🔥 差异代谢物多维热图工作台 (支持全组别比较与多类型热图切换)")
        
        hm_ctrl_1, hm_ctrl_2, hm_ctrl_3 = st.columns([2, 2, 1.5])
        with hm_ctrl_1:
            hm_scope_label = st.radio(
                "1️⃣ 选择展示的样本/组别范围",
                [
                    "仅当前对比两组 (Case vs Control)",
                    "全部实验组别 (自动剔除 QC 样本) [推荐多组趋势/回调]",
                    "全部样本 (包含 QC 样本)"
                ],
                index=0,
                key="hm_scope_radio"
            )
        with hm_ctrl_2:
            hm_type_label = st.radio(
                "2️⃣ 选择热图类型 (标准论文图表)",
                [
                    "1. 单样本明细热图 (按组别物理顺序排列)",
                    "2. 双向无监督聚类热图 (开启样本与代谢物双向聚类树)",
                    "3. 各组均值汇总热图 (Group-Averaged, 每组一列)",
                    "4. 差异代谢物自相关热图 (Pearson 共调控相关性矩阵)"
                ],
                index=0,
                key="hm_type_radio"
            )
        with hm_ctrl_3:
            hm_top_n = st.slider("展示 Top 差异代谢物数", 5, 100, 40, step=5, key="hm_top_n_slider")
            hm_show_mode_bar = st.checkbox("左侧显示采集模式颜色条", value=True, key="hm_mode_bar_chk")
            hm_show_vals = st.checkbox("在色块内标注数值", value=False, key="hm_annot_chk")

        scope_map = {
            "仅当前对比两组 (Case vs Control)": "pairwise",
            "全部实验组别 (自动剔除 QC 样本) [推荐多组趋势/回调]": "all_no_qc",
            "全部样本 (包含 QC 样本)": "all_with_qc"
        }
        type_map = {
            "1. 单样本明细热图 (按组别物理顺序排列)": "sample_sorted",
            "2. 双向无监督聚类热图 (开启样本与代谢物双向聚类树)": "sample_clustered",
            "3. 各组均值汇总热图 (Group-Averaged, 每组一列)": "group_mean",
            "4. 差异代谢物自相关热图 (Pearson 共调控相关性矩阵)": "correlation"
        }
        cur_scope_code = scope_map[hm_scope_label]
        cur_type_code = type_map[hm_type_label]

        png_bytes, pdf_bytes, hm_err = generate_custom_heatmap(
            res['df_all_unscaled'], res['out_df'], res['group_col'], res['sel_grps'],
            scope_mode=cur_scope_code, hm_type=cur_type_code,
            top_n=hm_top_n, show_row_mode=hm_show_mode_bar, show_cell_values=hm_show_vals
        )

        if png_bytes:
            dl_col1, dl_col2, _ = st.columns([1.5, 1.5, 3])
            clean_fname_base = f"Heatmap_{cur_type_code}_{cur_scope_code}_{safe_case_str}_vs_{safe_ctrl_str}"
            with dl_col1:
                st.download_button(
                    label="📸 下载 300 DPI 高清热图 (.png)",
                    data=png_bytes,
                    file_name=f"{clean_fname_base}.png",
                    mime="image/png",
                    type="primary",
                    key=f"dl_hm_png_{cur_type_code}_{cur_scope_code}_{hm_top_n}_{hm_show_mode_bar}_{hm_show_vals}"
                )
            with dl_col2:
                st.download_button(
                    label="📄 下载出版级矢量热图 (.pdf)",
                    data=pdf_bytes,
                    file_name=f"{clean_fname_base}.pdf",
                    mime="application/pdf",
                    key=f"dl_hm_pdf_{cur_type_code}_{cur_scope_code}_{hm_top_n}_{hm_show_mode_bar}_{hm_show_vals}"
                )
            
            b64_inline = base64.b64encode(png_bytes).decode('utf-8')
            st.markdown(
                f'<div style="text-align:center; margin-top:15px;">'
                f'<img src="data:image/png;base64,{b64_inline}" style="max-width:92%; height:auto; border:1px solid #e0e0e0; border-radius:6px; box-shadow:0 2px 8px rgba(0,0,0,0.06);">'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            st.info(hm_err or "当前筛选阈值下无显著差异代谢物，无法生成热图。")
            
    with tabs[7]:
        show_cols = [c for c in ['Name', 'Source_Mode', 'Detected_Modes', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr'] if c in res['out_df'].columns]
        if res.get('use_mode_b') and res.get('mode_results'):
            m_names = list(res['mode_results'].keys())
            table_tabs = st.tabs(
                [f"🏆 跨模式去重总表 ({len(res['out_df'])} 个)"] +
                [f"🔬 {m} 独立统计结果 ({len(res['mode_results'][m]['out_df'])} 个)" for m in m_names]
            )
            with table_tabs[0]:
                st.markdown("#### 🏆 跨模式汇总去重后的显著差异标志物清单 (以最小 P 值模式为代表)")
                st.dataframe(res['out_df'][show_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), width='stretch')
                st.download_button("📥 导出跨模式去重总差异物表 (.csv)", res['out_df'][show_cols].to_csv(index=False).encode('utf-8-sig'), f"Merged_Biomarkers_{safe_case_str}_vs_{safe_ctrl_str}.csv", "text/csv", key="dl_merged_bio")
                
            for idx, m in enumerate(m_names):
                with table_tabs[idx + 1]:
                    m_out = res['mode_results'][m]['out_df']
                    m_stats = res['mode_results'][m]['stats_df']
                    m_cols = [c for c in ['Name', 'Source_Mode', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr', 'Sig'] if c in m_stats.columns]
                    st.markdown(f"#### 🔬 模式 `{m}` 显著差异代谢物清单 (共 {len(m_out)} 个)")
                    st.dataframe(m_out[m_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), width='stretch')
                    c_dl1, c_dl2 = st.columns(2)
                    safe_m = safe_filename_part(m)
                    with c_dl1:
                        st.download_button(f"📥 导出 [{m}] 显著差异代谢物 (.csv)", m_out[m_cols].to_csv(index=False).encode('utf-8-sig'), f"Biomarkers_{safe_m}_{safe_case_str}_vs_{safe_ctrl_str}.csv", "text/csv", key=f"dl_sig_{m}")
                    with c_dl2:
                        st.download_button(f"📥 导出 [{m}] 全特征完整统计表 (.csv)", m_stats[m_cols].to_csv(index=False).encode('utf-8-sig'), f"All_Stats_{safe_m}_{safe_case_str}_vs_{safe_ctrl_str}.csv", "text/csv", key=f"dl_all_{m}")
                    with st.expander(f"展开查看模式 `{m}` 全部 {len(m_stats)} 个清洗后特征的统计结果"):
                        st.dataframe(m_stats[m_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}), width='stretch')
        else:
            st.markdown(f"### 🏆 显著差异生物标志物清单 (共 {len(res['out_df'])} 个 / 清洗后背景特征 {len(res['stats_df'])} 个)")
            st.dataframe(res['out_df'][show_cols].style.format({"Log2_FC":"{:.2f}", "P_Value":"{:.3e}", "FDR":"{:.3e}", "VIP":"{:.2f}", "p_corr":"{:.2f}"}).background_gradient(subset=['VIP'], cmap="Reds"), width='stretch')
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.download_button("📥 导出显著差异代谢物清单 (.csv)", res['out_df'][show_cols].to_csv(index=False).encode('utf-8-sig'), f"Biomarkers_{safe_case_str}_vs_{safe_ctrl_str}.csv", "text/csv", key="dl_single_bio")
            with c_s2:
                all_cols = [c for c in ['Name', 'Source_Mode', 'Detected_Modes', 'Log2_FC', 'P_Value', 'FDR', 'VIP', 'p_corr', 'Sig'] if c in res['stats_df'].columns]
                st.download_button("📥 导出全特征完整统计表 (.csv)", res['stats_df'][all_cols].to_csv(index=False).encode('utf-8-sig'), f"All_Stats_{safe_case_str}_vs_{safe_ctrl_str}.csv", "text/csv", key="dl_single_all")
        
    with tabs[8]:
        c1, c2 = st.columns([1, 6])
        with c2:
            if res['fig_nomogram']: st.plotly_chart(res['fig_nomogram'], key="nomo_main")
            else: st.warning("⚠️ 显著差异代谢物不足 2 个或分组异常，无法构建列线图。")
            
    with tabs[9]:
        if 'filtered_db_df' in res and not res['filtered_db_df'].empty:
            st.download_button("📥 导出专属 MA 背景库", res['filtered_db_df'].to_csv(index=False, header=False, quoting=csv.QUOTE_ALL).encode('utf-8'), f"MA_Background_{safe_case_str}_{safe_ctrl_str}.csv", "text/csv", type="primary")
        c1, c2 = st.columns([1, 6])
        with c2:
            if res['pathway_df'].empty: st.warning("未能匹配到富集通路。")
            else:
                if res['fig_pathway']: st.plotly_chart(res['fig_pathway'], key="pw_main")
                st.dataframe(res['pathway_df'].drop(columns=['-Log10_P'], errors='ignore').style.format({"P_Value":"{:.3e}", "FDR":"{:.3e}", "Enrichment_Factor":"{:.2f}"}).background_gradient(subset=['P_Value'], cmap="Reds_r", vmin=0, vmax=0.05), width='stretch')
                
    with tabs[10]:
        if res['fig_network']: st.plotly_chart(res['fig_network'], key="net_main")
        else: st.info("没有找到通路与代谢物的有效映射。")
        
    with tabs[11]:
        c_rep1, c_rep2 = st.columns(2)
        with c_rep1: st.download_button("📥 离线网页报告 (.html)", res['html_report'].encode('utf-8'), f"Report_{safe_case_str}_{safe_ctrl_str}.html", "text/html")
        with c_rep2: st.download_button("📥 AI Prompt (.md)", res['prompt_md'].encode('utf-8'), f"Prompt_{safe_case_str}_{safe_ctrl_str}.md", "text/markdown")
