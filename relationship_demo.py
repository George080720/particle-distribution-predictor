import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt

st.title("粒子分布预测工具")
st.write("输入 DP（聚合度）和 Ca²⁺ 浓度，预测综合指数。")

# ========== 1. 读数据 + 训练模型（去掉缓存） ==========
df = pd.read_csv("relationship.csv")

X = df[['DP', 'Ca²⁺ concentration (mM)']].values
y = df['Composite Index'].values

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# ========== 2. 侧边栏：输入 ==========
st.sidebar.header("输入参数")
dp = st.sidebar.slider("DP（聚合度）", 20, 80, 54, step=1)
ca = st.sidebar.slider("Ca²⁺ 浓度 (mM)", 1, 5, 3, step=1)

# ========== 3. 预测 ==========
pred = model.predict([[dp, ca]])[0]

st.subheader("预测结果")
st.metric("综合指数", f"{pred:.2f}")

if pred > 0.4:
    st.success("分布质量：好（粒子分布较均匀）")
elif pred > 0.15:
    st.warning("分布质量：中等")
else:
    st.error("分布质量：差（粒子分布不均匀）")

# ========== 4. 模型信息 ==========
st.subheader("模型信息")

y_pred = model.predict(X)
r2 = r2_score(y, y_pred)
mae = mean_absolute_error(y, y_pred)

col1, col2 = st.columns(2)
with col1:
    st.metric("R²", f"{r2:.3f}")
with col2:
    st.metric("MAE", f"{mae:.3f}")

# ========== 5. 特征重要性 ==========
st.subheader("特征重要性")
importance = model.feature_importances_
fig, ax = plt.subplots()
ax.barh(['DP', 'Ca²⁺ concentration (mM)'], importance, color=['steelblue', 'coral'])
ax.set_xlabel('importance')
ax.set_xlim(0, 1)
for i, v in enumerate(importance):
    ax.text(v + 0.01, i, f"{v:.3f}", va='center')
st.pyplot(fig)

# ========== 6. 预测热图 ==========
st.subheader("不同 DP + Ca²⁺ 浓度下的综合指数预测")

dp_range = list(range(28, 73, 4))
ca_range = list(range(1, 6))

heatmap = np.zeros((len(dp_range), len(ca_range)))
for i, dp_val in enumerate(dp_range):
    for j, ca_val in enumerate(ca_range):
        heatmap[i, j] = model.predict([[dp_val, ca_val]])[0]

fig2, ax2 = plt.subplots(figsize=(6, 5))
im = ax2.imshow(heatmap, cmap='YlGn', aspect='auto', vmin=0, vmax=1)
ax2.set_xticks(range(len(ca_range)))
ax2.set_xticklabels(ca_range)
ax2.set_yticks(range(len(dp_range)))
ax2.set_yticklabels(dp_range)
ax2.set_xlabel("Ca²⁺ concentration (mM)")
ax2.set_ylabel("DP（聚合度）")
plt.colorbar(im, ax=ax2, label="Composite Index")

for i in range(len(dp_range)):
    for j in range(len(ca_range)):
        ax2.text(j, i, f"{heatmap[i,j]:.2f}", ha='center', va='center', fontsize=8)

st.pyplot(fig2)

# ========== 7. 数据预览 ==========
st.subheader("训练数据")
st.dataframe(df)
