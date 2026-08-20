"""
Android Games - Hit Game Predictor
Streamlit app that reproduces the EDA + preprocessing + feature engineering +
model training/comparison pipeline (Logistic Regression / Random Forest / XGBoost)
originally built in the notebook `knkm.ipynb`.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

st.set_page_config(page_title="Hit Game Predictor", layout="wide")

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def drop_if_exists(df, cols):
    existing = [c for c in cols if c in df.columns]
    return df.drop(columns=existing), existing


def plot_confusion_matrix(cm, title):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return fig


# ----------------------------------------------------------------------
# Sidebar - data loading
# ----------------------------------------------------------------------

st.sidebar.title("⚙️ الإعدادات")
uploaded_file = st.sidebar.file_uploader("ارفع ملف android_games_eda_ready.csv", type=["csv"])

st.title("🎮 Android Games — Hit Game Predictor")
st.caption("نسخة Streamlit تفاعلية من نفس بايبلاين النوت بوك: تنظيف بيانات → معالجة → موديلات → مقارنة")

if uploaded_file is None:
    st.info("⬅️ ارفع ملف الـ CSV من القائمة الجانبية عشان تبدأ.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

with st.expander("👀 نظرة عامة على البيانات الخام", expanded=True):
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("**Shape:**", df_raw.shape)
        st.write("**Columns:**", len(df_raw.columns))
    with c2:
        st.dataframe(df_raw.head(), use_container_width=True)

# ----------------------------------------------------------------------
# Step 1: Cleaning
# ----------------------------------------------------------------------

st.header("1️⃣ تنظيف البيانات")

df = df_raw.copy()

id_like_cols = ['game_id', 'game_name', 'package_name', 'row_checksum_id']
df, dropped_id = drop_if_exists(df, id_like_cols)

n_dupes = df.duplicated().sum()
df.drop_duplicates(inplace=True)

if 'soft_launch_date' in df.columns:
    df['soft_launch_date'] = df['soft_launch_date'].fillna('No Soft Launch')

cols_to_drop_step3 = [
    'featured_start_date', 'featured_end_date', 'featured_duration_days',
    'event_theme', 'multiplayer_support'
]
df, dropped_step3 = drop_if_exists(df, cols_to_drop_step3)

cols_to_drop_step4 = [
    'store_platform', 'post_30d_revenue_usd', 'post_30d_rating_count',
    'downloads', 'total_revenue_usd', 'release_date',
    'soft_launch_date', 'last_update_date'
]
df, dropped_step4 = drop_if_exists(df, cols_to_drop_step4)

c1, c2, c3 = st.columns(3)
c1.metric("أعمدة ID اتحذفت", len(dropped_id))
c2.metric("صفوف مكررة اتحذفت", int(n_dupes))
c3.metric("أعمدة تانية اتحذفت", len(dropped_step3) + len(dropped_step4))

st.write("**Shape بعد التنظيف:**", df.shape)

with st.expander("تفاصيل الأعمدة المحذوفة"):
    st.write("ID-like:", dropped_id)
    st.write("Featured/event cols:", dropped_step3)
    st.write("Leakage/platform cols:", dropped_step4)

st.subheader("القيم الفارغة (Nulls) المتبقية")
nulls = df.isnull().sum()
st.dataframe(nulls[nulls > 0].rename("null_count"))

target_col = "is_hit_game"
if target_col not in df.columns:
    st.error(f"عمود الهدف '{target_col}' مش موجود في الملف اللي رفعته. الرجاء التأكد من ملف البيانات.")
    st.stop()

# ----------------------------------------------------------------------
# Step 2: Train/test split
# ----------------------------------------------------------------------

st.header("2️⃣ تقسيم البيانات (Train / Test)")

test_size = st.slider("نسبة بيانات الاختبار (Test size)", 0.1, 0.4, 0.2, 0.05)

X = df.drop(columns=[target_col])
y = df[target_col]

st.write("**توازن الفئات (Class balance) في is_hit_game:**")
st.dataframe(y.value_counts(normalize=True).rename("proportion"))

fig, ax = plt.subplots(figsize=(4, 3))
sns.countplot(x=y, ax=ax)
ax.set_xlabel("is_hit_game")
st.pyplot(fig)
plt.close(fig)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42, stratify=y
)

c1, c2 = st.columns(2)
c1.write(f"X_train: {X_train.shape}")
c2.write(f"X_test: {X_test.shape}")

# ----------------------------------------------------------------------
# Step 3: Missing value imputation (numeric)
# ----------------------------------------------------------------------

st.header("3️⃣ معالجة القيم الفارغة")

fill_candidates = ['marketing_spend_usd', 'cpi_usd', 'arppu_usd']
fill_cols = [c for c in fill_candidates if c in X_train.columns]

for col in fill_cols:
    train_median = X_train[col].median()
    X_train[col] = X_train[col].fillna(train_median)
    X_test[col] = X_test[col].fillna(train_median)

st.write(f"تم ملء القيم الفارغة في الأعمدة: {fill_cols} بالـ median الخاص بالـ train.")
st.write("Nulls متبقية في X_train:", int(X_train.isnull().sum().sum()))
st.write("Nulls متبقية في X_test:", int(X_test.isnull().sum().sum()))

# ----------------------------------------------------------------------
# Step 4: developer_name -> frequency feature
# ----------------------------------------------------------------------

if 'developer_name' in X_train.columns:
    dev_counts = X_train['developer_name'].value_counts()
    X_train['developer_game_count'] = X_train['developer_name'].map(dev_counts)
    X_test['developer_game_count'] = X_test['developer_name'].map(dev_counts).fillna(0)
    X_train = X_train.drop(columns=['developer_name'])
    X_test = X_test.drop(columns=['developer_name'])

# ----------------------------------------------------------------------
# Step 5: Outlier capping (IQR)
# ----------------------------------------------------------------------

st.header("4️⃣ التعامل مع القيم الشاذة (Outliers - IQR)")

binary_like = ['was_featured_on_store', 'is_weekend_release', 'has_seasonal_event',
               'cross_platform_available', 'cloud_save_support']
zero_inflated = ['price_usd']
num_col = X_train.select_dtypes(include=[np.number]).columns
num_col_to_cap = [c for c in num_col if c not in binary_like and c not in zero_inflated]

outlier_summary = {}
for col in num_col_to_cap:
    Q1 = X_train[col].quantile(0.25)
    Q3 = X_train[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = X_train[(X_train[col] < lower_bound) | (X_train[col] > upper_bound)]
    outlier_summary[col] = {
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'lower_bound': lower_bound, 'upper_bound': upper_bound,
        'num_outliers': len(outliers)
    }

outlier_df = pd.DataFrame(outlier_summary).T
with st.expander("جدول تفاصيل الـ Outliers لكل عمود"):
    st.dataframe(outlier_df, use_container_width=True)

do_cap = st.checkbox("طبّق الـ Capping على القيم الشاذة (موصى به، زي النوت بوك)", value=True)

if do_cap:
    for col in num_col_to_cap:
        Q1 = X_train[col].quantile(0.25)
        Q3 = X_train[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_train[col] = np.clip(X_train[col], lower_bound, upper_bound)
        X_test[col] = np.clip(X_test[col], lower_bound, upper_bound)
    st.success("تم تطبيق الـ capping.")

# ----------------------------------------------------------------------
# Step 6: Log transform + scaling
# ----------------------------------------------------------------------

st.header("5️⃣ التحويلات: Log Transform + Scaling")

log_cols_candidates = [
    'downloads', 'active_users_30d', 'rating_count', 'review_count',
    'marketing_spend_usd', 'total_revenue_usd', 'wishlist_prelaunch',
    'post_30d_revenue_usd', 'post_30d_rating_count'
]
log_cols = [c for c in log_cols_candidates if c in X_train.columns]

for c in log_cols:
    X_train[c] = np.log1p(X_train[c])
    X_test[c] = np.log1p(X_test[c])

st.write(f"أعمدة تم تطبيق log1p عليها: {log_cols}")

standard_cols_candidates = [
    'price_usd', 'retention_day1_pct', 'retention_day7_pct', 'retention_day30_pct',
    'avg_session_minutes', 'avg_daily_sessions', 'crash_rate_pct', 'apk_size_mb',
    'update_frequency_days', 'rating_avg', 'cpi_usd', 'conversion_to_payer_pct',
    'ad_impressions_per_user', 'ad_revenue_share_pct', 'iap_revenue_share_pct',
    'arpu_usd', 'arppu_usd', 'days_since_release', 'days_since_update',
    'days_since_soft_launch', 'featured_gap_days', 'developer_game_count'
]
standard_cols = [c for c in standard_cols_candidates if c in X_train.columns]

scaler = StandardScaler()
if standard_cols:
    X_train[standard_cols] = scaler.fit_transform(X_train[standard_cols])
    X_test[standard_cols] = scaler.transform(X_test[standard_cols])

st.write(f"أعمدة تم عمل StandardScaler لها: {standard_cols}")

# ----------------------------------------------------------------------
# Step 7: Categorical frequency encoding
# ----------------------------------------------------------------------

st.header("6️⃣ ترميز الأعمدة الفئوية (Frequency Encoding)")

cat_cols_candidates = [
    'genre', 'sub_genre', 'monetization_model', 'contains_ads',
    'has_in_app_purchases', 'age_rating', 'art_style', 'engine_used',
    'region_focus', 'publisher_tier', 'country_code_primary'
]
cat_cols = [c for c in cat_cols_candidates if c in X_train.columns]

for col in cat_cols:
    counts = X_train[col].value_counts()
    X_train[col + '_count'] = X_train[col].map(counts)
    X_test[col + '_count'] = X_test[col].map(counts).fillna(0)

X_train = X_train.drop(columns=cat_cols)
X_test = X_test.drop(columns=cat_cols)

st.write(f"أعمدة فئوية تم تحويلها لـ frequency count: {cat_cols}")

# label encode target
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# bool -> int
bool_features = X_train.select_dtypes(include="bool").columns
for col in bool_features:
    X_train[col] = X_train[col].astype(int)
    X_test[col] = X_test[col].astype(int)

st.write("**Shape النهائي بعد الـ Feature Engineering:**")
c1, c2 = st.columns(2)
c1.write(f"X_train: {X_train.shape}")
c2.write(f"X_test: {X_test.shape}")

with st.expander("عرض X_train بعد المعالجة"):
    st.dataframe(X_train.head(), use_container_width=True)

# ----------------------------------------------------------------------
# Step 8: Correlation + multicollinearity + weak-feature removal
# ----------------------------------------------------------------------

st.header("7️⃣ اختيار الخصائص (Feature Selection)")

new_data = pd.concat(
    [
        X_train.select_dtypes(include="number").reset_index(drop=True),
        pd.Series(y_train_enc, name=target_col).reset_index(drop=True),
    ],
    axis=1,
)
corr = new_data.corr()

with st.expander("🔥 خريطة الارتباط (Correlation Heatmap)"):
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    plt.close(fig)

corr_target = abs(corr[target_col]).drop(target_col)

multi_thresh = st.slider("حد التعدد الخطي (multicollinearity) بين الخصائص", 0.5, 0.95, 0.7, 0.05)
weak_thresh = st.slider("حد ضعف الارتباط مع الهدف (weak correlation)", 0.0, 0.1, 0.02, 0.01)

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
multi_features = [c for c in upper.columns if c != target_col and any(upper[c] > multi_thresh)]

weak_features = list(corr_target[corr_target <= weak_thresh].index)

drop_final = list(dict.fromkeys(multi_features + weak_features))  # unique, preserve order
drop_final = [c for c in drop_final if c in X_train.columns]

c1, c2 = st.columns(2)
with c1:
    st.write(f"**خصائص هيتم حذفها بسبب تعدد خطي (> {multi_thresh}):**")
    st.write(multi_features if multi_features else "لا يوجد")
with c2:
    st.write(f"**خصائص هيتم حذفها بسبب ارتباط ضعيف (≤ {weak_thresh}):**")
    st.write(weak_features if weak_features else "لا يوجد")

X_train_fs = X_train.drop(columns=drop_final, errors='ignore')
X_test_fs = X_test.drop(columns=drop_final, errors='ignore')

st.write(f"**عدد الخصائص النهائي:** {X_train_fs.shape[1]}")

with st.expander("📊 Mutual Information Scores"):
    mi_scores = mutual_info_regression(X_train_fs, y_train_enc, random_state=42)
    mi = pd.Series(mi_scores, index=X_train_fs.columns).sort_values(ascending=False)
    st.dataframe(mi.rename("MI score"))

# ----------------------------------------------------------------------
# Step 9: Model training & comparison
# ----------------------------------------------------------------------

st.header("8️⃣ تدريب ومقارنة الموديلات")

available_models = ["Logistic Regression", "Random Forest"]
if XGB_AVAILABLE:
    available_models.append("XGBoost")
else:
    st.warning("مكتبة xgboost مش متاحة في البيئة دي — هيتم استبعادها. تقدر تضيفها بـ `pip install xgboost`.")

chosen_models = st.multiselect("اختار الموديلات اللي عايز تدرّبها", available_models, default=available_models)
threshold = st.slider("Classification threshold", 0.1, 0.9, 0.5, 0.05)

run = st.button("🚀 درّب الموديلات")

if run:
    if not chosen_models:
        st.warning("اختار موديل واحد على الأقل.")
        st.stop()

    results = []
    trained_models = {}

    scale_pos_weight = (y_train_enc == 0).sum() / max((y_train_enc == 1).sum(), 1)

    with st.spinner("جاري تدريب الموديلات..."):
        if "Logistic Regression" in chosen_models:
            lr = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
            lr.fit(X_train_fs, y_train_enc)
            trained_models["Logistic Regression"] = lr

        if "Random Forest" in chosen_models:
            rf = RandomForestClassifier(
                n_estimators=200, max_depth=20, class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            rf.fit(X_train_fs, y_train_enc)
            trained_models["Random Forest"] = rf

        if "XGBoost" in chosen_models and XGB_AVAILABLE:
            xgb = XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.1,
                scale_pos_weight=scale_pos_weight, eval_metric='logloss',
                random_state=42, n_jobs=-1
            )
            xgb.fit(X_train_fs, y_train_enc)
            trained_models["XGBoost"] = xgb

    tabs = st.tabs(list(trained_models.keys()))

    for tab, (name, model) in zip(tabs, trained_models.items()):
        with tab:
            y_proba_train = model.predict_proba(X_train_fs)[:, 1]
            y_proba_test = model.predict_proba(X_test_fs)[:, 1]
            y_pred_train = (y_proba_train >= threshold).astype(int)
            y_pred_test = (y_proba_test >= threshold).astype(int)

            metrics = {
                'Model': name,
                'Threshold': threshold,
                'Accuracy': accuracy_score(y_test_enc, y_pred_test),
                'Precision': precision_score(y_test_enc, y_pred_test, zero_division=0),
                'Recall': recall_score(y_test_enc, y_pred_test, zero_division=0),
                'F1': f1_score(y_test_enc, y_pred_test, zero_division=0),
            }
            results.append(metrics)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['Accuracy']:.3f}")
            c2.metric("Precision", f"{metrics['Precision']:.3f}")
            c3.metric("Recall", f"{metrics['Recall']:.3f}")
            c4.metric("F1", f"{metrics['F1']:.3f}")

            cc1, cc2 = st.columns(2)
            with cc1:
                st.text("Classification Report (Test):")
                st.text(classification_report(y_test_enc, y_pred_test, zero_division=0))
            with cc2:
                cm = confusion_matrix(y_test_enc, y_pred_test)
                fig = plot_confusion_matrix(cm, f"{name} — Test Confusion Matrix")
                st.pyplot(fig)
                plt.close(fig)

    st.subheader("📈 مقارنة الموديلات")
    results_df = pd.DataFrame(results).sort_values(by='F1', ascending=False).reset_index(drop=True)
    st.dataframe(results_df.round(4), use_container_width=True)

    best_model_name = results_df.iloc[0]['Model']
    st.success(f"🏆 أفضل موديل حسب F1-score: **{best_model_name}**")

    fig, ax = plt.subplots(figsize=(10, 5))
    results_df.set_index('Model')[['Accuracy', 'Precision', 'Recall', 'F1']].plot(
        kind='bar', ax=ax, ylim=(0, 1)
    )
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', fontsize=8, padding=2)
    ax.set_title("Model Comparison")
    ax.set_ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # store in session for optional prediction section
    st.session_state["trained_models"] = trained_models
    st.session_state["feature_cols"] = list(X_train_fs.columns)

else:
    st.info("اضبط الإعدادات فوق واضغط زرار 'درّب الموديلات' عشان تشوف النتائج.")