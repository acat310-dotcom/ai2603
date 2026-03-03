import streamlit as st
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# 設定頁面資訊
st.set_page_config(page_title="🍷 酒類分類預測系統", layout="wide")

# 套用自定義 CSS 以增加現代感
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        height: 3em;
        background-color: #722f37;
        color: white;
    }
    .stSidebar {
        background-color: #f1f3f5;
    }
    h1 {
        color: #722f37;
    }
    </style>
    """, unsafe_allow_html=True)

# 標題
st.title("🍷 酒類分類預測系統")

# 1. 側邊欄設定
st.sidebar.header("⚙️ 設定與資訊")

# 模型選擇下拉選單
model_name = st.sidebar.selectbox(
    "選擇模型",
    ("KNN", "羅吉斯迴歸", "XGBoost", "隨機森林")
)

# 2. 顯示「酒類」資料集資訊
st.sidebar.markdown("---")
st.sidebar.subheader("ℹ️ 資料集資訊：酒類 (Wine)")
st.sidebar.info("""
Wine 資料集包含三種不同酒類的化學成分分析。
- 特徵數量：13 (如酒精濃度、蘋果酸等)
- 分類目標：3 類酒
- 樣本人數：178 筆
""")

# 載入資料集
@st.cache_data
def load_data():
    wine = datasets.load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine

df, wine_raw = load_data()

# 3. 右側 Main 區
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 資料集前 5 筆")
    st.dataframe(df.head())

with col2:
    st.subheader("📊 特徵統計值")
    st.write(df.describe())

# 4. 預測功能
st.markdown("---")
st.header("🚀 模型訓練與預測")

if st.button("開始進行預測"):
    with st.spinner('模型訓練中...'):
        # 準備資料
        X = df.drop('target', axis=1)
        y = df['target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 根據選擇的模型初始化
        if model_name == "KNN":
            model = KNeighborsClassifier(n_neighbors=5)
        elif model_name == "羅吉斯迴歸":
            model = LogisticRegression(max_iter=5000)
        elif model_name == "XGBoost":
            model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        elif model_name == "隨機森林":
            model = RandomForestClassifier(n_estimators=100)

        # 訓練模型
        model.fit(X_train, y_train)
        
        # 進行預測
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # 顯示結果
        st.success(f"### 使用模型：{model_name}")
        
        # 使用 Metrics 展示結果
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("預測準確度", f"{acc:.2%}")
        
        st.write("#### 前 10 筆預測結果 vs 實際結果")
        results_compare = pd.DataFrame({
            '實際值': [wine_raw.target_names[i] for i in y_test[:10]],
            '預測值': [wine_raw.target_names[i] for i in y_pred[:10]]
        })
        st.table(results_compare)

st.markdown("---")
st.caption("Developed by Antigravity AI Engine")
