import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

def update_layout_square(fig, title="", x_title="", y_title=""):
    """标准化 Plotly 图表为正方形白底样式"""
    fig.update_layout(
        template="simple_white", width=600, height=600, 
        title={'text': title, 'y':0.95, 'x':0.5, 'xanchor': 'center'}, 
        xaxis=dict(title=x_title, showline=True, linewidth=2, mirror=True), 
        yaxis=dict(title=y_title, showline=True, linewidth=2, mirror=True), 
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.15)
    )
    return fig

def get_ellipse_coordinates(x, y, std_mult=2):
    """计算 95% 置信椭圆的坐标"""
    if len(x) < 3: return None, None
    mean_x, mean_y = np.mean(x), np.mean(y)
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:,order]
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
    width, height = 2 * std_mult * np.sqrt(vals)
    t = np.linspace(0, 2*np.pi, 100)
    ell_x = width/2 * np.cos(t); ell_y = height/2 * np.sin(t)
    rad = np.radians(theta)
    R = np.array([[np.cos(rad), -np.sin(rad)], [np.sin(rad), np.cos(rad)]])
    ell_coords = np.dot(R, np.array([ell_x, ell_y]))
    return ell_coords[0] + mean_x, ell_coords[1] + mean_y

def plot_nomogram(df_sub, features, feature_names, group_col, case_name):
    """构建逻辑回归并使用底层 Plotly 画线实现列线图"""
    X = df_sub[features].values
    y = np.where(df_sub[group_col] == case_name, 1, 0)
    
    if len(np.unique(y)) < 2: return None
    
    clf = LogisticRegression(C=100.0, solver='lbfgs', max_iter=1000)
    clf.fit(X, y)
    betas = clf.coef_[0]
    beta0 = clf.intercept_[0]
    
    L_min, L_max, L_ranges = [], [], []
    for i in range(len(features)):
        v_min, v_max = np.min(X[:, i]), np.max(X[:, i])
        L_v1, L_v2 = betas[i]*v_min, betas[i]*v_max
        L_min.append(min(L_v1, L_v2))
        L_max.append(max(L_v1, L_v2))
        L_ranges.append(abs(L_v1 - L_v2))
        
    L_max_range = max(L_ranges) if max(L_ranges) > 0 else 1e-5
    S = 100.0 / L_max_range
    
    fig = go.Figure()
    y_labels = ['Risk Prob', 'Total Points'] + [f[:15]+"..." if len(f)>15 else f for f in feature_names[::-1]] + ['Points']
    y_vals = list(range(len(y_labels)))
    
    yp = y_vals[-1]
    fig.add_trace(go.Scatter(x=[0, 100], y=[yp, yp], mode='lines', line=dict(color='black', width=2), showlegend=False))
    for pt in range(0, 101, 10):
        fig.add_trace(go.Scatter(x=[pt, pt], y=[yp, yp+0.15], mode='lines', line=dict(color='black', width=2), showlegend=False))
        fig.add_annotation(x=pt, y=yp+0.4, text=str(pt), showarrow=False, font=dict(size=12))
        
    TP_max = 0
    for idx, _ in enumerate(features[::-1]):
        i = len(features) - 1 - idx
        yf = y_vals[2 + idx]
        pts_max = S * L_ranges[i]
        TP_max += pts_max
        fig.add_trace(go.Scatter(x=[0, pts_max], y=[yf, yf], mode='lines', line=dict(color='black', width=2), showlegend=False))
        v_min, v_max = np.min(X[:, i]), np.max(X[:, i])
        ticks = np.linspace(v_min, v_max, 5)
        for tv in ticks:
            L_tv = betas[i] * tv
            pt_tv = S * (L_tv - L_min[i])
            fig.add_trace(go.Scatter(x=[pt_tv, pt_tv], y=[yf, yf+0.15], mode='lines', line=dict(color='black', width=2), showlegend=False))
            fig.add_annotation(x=pt_tv, y=yf+0.4, text=f"{tv:.2f}", showarrow=False, font=dict(size=11))
            
    ytp = y_vals[1]
    fig.add_trace(go.Scatter(x=[0, TP_max], y=[ytp, ytp], mode='lines', line=dict(color='black', width=2), showlegend=False))
    for pt in range(0, int(TP_max)+1, max(10, int(TP_max//10))):
        fig.add_trace(go.Scatter(x=[pt, pt], y=[ytp, ytp+0.15], mode='lines', line=dict(color='black', width=2), showlegend=False))
        fig.add_annotation(x=pt, y=ytp+0.4, text=str(pt), showarrow=False, font=dict(size=12))
        
    yprob = y_vals[0]
    def tp_to_prob(tp):
        logit = np.clip(beta0 + (tp / S) + sum(L_min), -20, 20)
        return 1 / (1 + np.exp(-logit))
    def prob_to_tp(p):
        return S * (np.log(p / (1 - p)) - beta0 - sum(L_min))
        
    min_p, max_p = tp_to_prob(0), tp_to_prob(TP_max)
    probs_to_plot = [0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99]
    valid_probs = [p for p in probs_to_plot if min_p <= p <= max_p]
    
    fig.add_trace(go.Scatter(x=[0, TP_max], y=[yprob, yprob], mode='lines', line=dict(color='black', width=2), showlegend=False))
    for p in valid_probs:
        tp = prob_to_tp(p)
        fig.add_trace(go.Scatter(x=[tp, tp], y=[yprob, yprob+0.15], mode='lines', line=dict(color='black', width=2), showlegend=False))
        fig.add_annotation(x=tp, y=yprob+0.4, text=f"{p:.2f}", showarrow=False, font=dict(size=11))
        
    fig.update_layout(
        yaxis=dict(tickvals=y_vals, ticktext=y_labels, showgrid=False, zeroline=False, tickfont=dict(size=13, color='black')),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white', height=250 + 70*len(features),
        margin=dict(l=150, r=50, t=50, b=50),
        title={'text': "Diagnostic Nomogram (Logistic Regression)", 'x':0.5, 'xanchor': 'center'}
    )
    return fig
