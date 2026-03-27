import pandas as pd

def generate_offline_html(case, ctrl, feats, p_th, fc_th, norm_m, scale_m, R2Y, Q2, b_q2, 
                          out_df, pathway_df, fig_opls, fig_perm, fig_splot, fig_vol, fig_pca, 
                          hm_base64, fig_nomogram, fig_pathway, fig_network, 
                          vip_show_num, pw_show_num, nomo_num):
    
    js_added = [False] 
    def get_html_plot(fig):
        if fig is not None:
            if not js_added[0]:
                js_added[0] = True
                return fig.to_html(full_html=False, include_plotlyjs=True)
            else:
                return fig.to_html(full_html=False, include_plotlyjs=False)
        return "<p style='color:red;'>未生成该图表</p >"

    # 🟢 完美解决对齐问题：在转成 HTML 前，过滤掉 Hits 为 0 的背景通路
    pw_html = "<p>未进行通路富集分析或无显著命中。</p >"
    if not pathway_df.empty and 'Hits' in pathway_df.columns:
        valid_pw_df = pathway_df[pathway_df['Hits'] > 0].head(pw_show_num)
        if not valid_pw_df.empty:
            pw_html = valid_pw_df[['Pathway', 'Total_in_Pathway', 'Hits', 'Enrichment_Factor', 'P_Value']].to_html(index=False, float_format="%.4f")
    
    html_report = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>代谢组学综合分析报告 | {case} vs {ctrl}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px auto; max-width: 1100px; color: #333; line-height: 1.6; background-color: #f4f7f6; }}
            .container {{ background-color: #fff; padding: 40px; border-radius: 10px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; text-align: center; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #2980b9; margin-top: 40px; border-left: 4px solid #2980b9; padding-left: 10px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; text-align: left; }}
            th, td {{ border: 1px solid #ddd; padding: 10px; }}
            th {{ background-color: #f8f9fa; color: #2c3e50; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .plot-box {{ margin: 30px 0; padding: 15px; border: 1px solid #eee; border-radius: 8px; background: #fafafa; text-align: center; }}
            .metric-container {{ display: flex; justify-content: space-around; background: #eef2f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
            .metric {{ text-align: center; }}
            .metric-title {{ font-size: 14px; color: #7f8c8d; text-transform: uppercase; }}
            .metric-value {{ font-size: 28px; font-weight: bold; color: #e74c3c; }}
        </style>
    </head>
    <body>
    <div class="container">
        <h1>代谢组学综合分析报告 (SIMCA 规范)</h1>
        
        <h2>1. 实验项目与参数</h2>
        <ul>
            <li><b>对比组别:</b> <code>{case}</code> (实验组) vs <code>{ctrl}</code> (对照组)</li>
            <li><b>数据规模:</b> 鉴定出 {len(feats)} 个特征</li>
            <li><b>筛选标准:</b> VIP > 1.0, P-value < {p_th}, |Log2 FC| > {fc_th}</li>
            <li><b>预处理策略:</b> {norm_m} + {scale_m} Scaling</li>
        </ul>
        
        <h2>2. OPLS-DA 模型评估</h2>
        <div class="metric-container">
            <div class="metric"><div class="metric-title">R²Y (模型解释率)</div><div class="metric-value">{R2Y:.3f}</div></div>
            <div class="metric"><div class="metric-title">Q² (模型预测率)</div><div class="metric-value">{Q2:.3f}</div></div>
            <div class="metric"><div class="metric-title">Q² 置换检验截距</div><div class="metric-value">{b_q2:.3f}</div></div>
        </div>
        <p style="text-align: center;"><b>结论:</b> {"该 OPLS-DA 模型分离度极佳且未发生过拟合，预测结果高度可靠。" if (b_q2 < 0.05 and Q2 > 0.5) else "模型分离度一般或存在轻微过拟合，提示两组间代谢差异可能不显著。"}</p >
        
        <h2>3. 核心差异代谢物清单 (Top {vip_show_num} Biomarkers)</h2>
        {out_df[['Name', 'Log2_FC', 'P_Value', 'VIP', 'p_corr']].head(vip_show_num).to_html(index=False, float_format="%.3f")}
        
        <h2>4. KEGG 代谢通路富集分析 (Top {pw_show_num})</h2>
        {pw_html}
        
        <h2>5. 统计与多维可视化图表</h2>
        <p><i>注：本报告为纯离线交互版。图表支持鼠标悬停、框选缩放，点击图表右上角相机图标 📷 即可下载透明底色高清图片。</i></p >
        
        <div class="plot-box"><h3>(1) OPLS-DA 得分图</h3>{get_html_plot(fig_opls)}</div>
        <div class="plot-box"><h3>(2) 置换检验 (Permutation Test)</h3>{get_html_plot(fig_perm)}</div>
        <div class="plot-box"><h3>(3) S-Plot</h3>{get_html_plot(fig_splot)}</div>
        <div class="plot-box"><h3>(4) 火山图</h3>{get_html_plot(fig_vol)}</div>
        <div class="plot-box"><h3>(5) PCA 宏观质控得分图</h3>{get_html_plot(fig_pca)}</div>
    """
    
    if hm_base64:
        clean_b64 = hm_base64.replace('\n', '').replace('\r', '').strip()
        html_report += f'''
        <div class="plot-box">
            <h3>(6) Top 50 差异代谢物聚类热图</h3>
            <img src="data:image/png;base64,{clean_b64}" style="max-width:100%; border:1px solid #ccc;"/>
        </div>
        '''
        
    if fig_nomogram is not None:
        html_report += f'''<div class="plot-box"><h3>(7) 诊断预测列线图 (Top {nomo_num})</h3>{get_html_plot(fig_nomogram)}</div>'''
        
    if fig_pathway is not None:
        html_report += f'''<div class="plot-box"><h3>(8) KEGG 通路富集气泡图</h3>{get_html_plot(fig_pathway)}</div>'''
        
    if fig_network is not None:
        html_report += f'''<div class="plot-box"><h3>(9) 代谢重编程机制网络图</h3>{get_html_plot(fig_network)}</div>'''
        
    html_report += """
    </div>
    </body>
    </html>
    """
    return html_report

def generate_ai_prompt(case, ctrl, norm_m, scale_m, R2Y, Q2, b_q2, p_th, fc_th, out_df, pathway_df):
    num_up = len(out_df[out_df['Log2_FC'] > 0])
    num_down = len(out_df[out_df['Log2_FC'] < 0])
    top_mets_str = out_df[['Name', 'Log2_FC', 'P_Value', 'VIP']].head(15).to_markdown(index=False) if not out_df.empty else "无显著差异物"
    
    pw_str = "无显著富集通路"
    if not pathway_df.empty and 'Hits' in pathway_df.columns:
        sig_pws = pathway_df[(pathway_df['P_Value'] < 0.05) & (pathway_df['Hits'] > 0)].head(10)
        if not sig_pws.empty: pw_str = sig_pws[['Pathway', 'Hits', 'P_Value']].to_markdown(index=False)
    
    prompt_md = f"""请作为一名资深的生物信息学和代谢组学专家，根据以下我提供的代谢组学数据分析结果，帮我撰写一篇英文科研论文的 **Results（结果）** 和 **Discussion（讨论）** 部分。

### 🔬 1. 实验参数与模型质控
- **对比组别**: {case} (Case) vs {ctrl} (Control)
- **预处理与缩放**: {norm_m} + {scale_m} Scaling
- **OPLS-DA 模型评估**: R²Y = {R2Y:.3f}, Q² = {Q2:.3f}, 置换检验 Q² 截距 = {b_q2:.3f} (模型{"稳健且未过拟合" if (b_q2<0.05 and Q2>0.5) else "预测能力一般"})。

### 🧬 2. 差异生物标志物 (Biomarkers)
- 筛选阈值: VIP > 1.0 且 P-value < {p_th}。
- 整体情况: 共找到 {num_up + num_down} 个标志物，其中 {num_up} 个在 {case} 组中显著上调，{num_down} 个显著下调。
- **Top 15 核心标志物清单**:
{top_mets_str}

### 🕸️ 3. KEGG 代谢通路富集 (P < 0.05)
{pw_str}

---
### 📝 撰写要求：
1. **Results 部分**：总结 OPLS-DA 模型的分离情况和置换检验结果；描述差异代谢物的整体分布；客观描述上述显著富集的 KEGG 通路。
2. **Discussion 部分**：结合上述查出的 Top 标志物和关键通路，查阅最新生化医学文献，深入探讨 {case} 组相对于 {ctrl} 组发生这些代谢网络改变的生理/病理机制，以及潜在的临床指导意义。
"""
    return prompt_md
