import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

DATA = os.environ.get("KUAIRAND_DATA_DIR")
if not DATA:
    raise RuntimeError("请设置环境变量 KUAIRAND_DATA_DIR，指向本地 KuaiRand-1K/data 目录")

std1 = pd.read_csv(f"{DATA}/log_standard_4_08_to_4_21_1k.csv")
std2 = pd.read_csv(f"{DATA}/log_standard_4_22_to_5_08_1k.csv")
rand = pd.read_csv(f"{DATA}/log_random_4_22_to_5_08_1k.csv")

print("=== 随机曝光日志诊断 ===")
print(f"原始行数: {len(rand)}")
print(f"完全重复行数: {rand.duplicated().sum()}")
print(f"duration_ms<=0 行数: {(rand['duration_ms'] <= 0).sum()}")

rand = rand.drop_duplicates()
rand_quarantined = rand[rand['duration_ms'] <= 0]
rand_clean = rand[rand['duration_ms'] > 0].copy()
rand_clean['source'] = 'random'
print(f"清洗后剩余: {len(rand_clean)}")

std1['source'] = 'standard'
std2['source'] = 'standard'
std = pd.concat([std1, std2], ignore_index=True)
std = std.drop_duplicates(subset=[c for c in std.columns if c != 'source'])
std = std[std['duration_ms'] > 0]

full = pd.concat([std, rand_clean], ignore_index=True)
full = full.sort_values(['user_id', 'time_ms']).reset_index(drop=True)

ts_group = full.groupby(['user_id', 'time_ms'])['long_view'].agg(['sum', 'size']).reset_index()
ts_group.columns = ['user_id', 'time_ms', 'lv_sum', 'row_count']
ts_group = ts_group.sort_values(['user_id', 'time_ms'])

ts_group['cum_count_incl'] = ts_group.groupby('user_id')['row_count'].cumsum()
ts_group['hist_interaction_count'] = ts_group.groupby('user_id')['cum_count_incl'].shift(1).fillna(0)

ts_group['cum_lv_incl'] = ts_group.groupby('user_id')['lv_sum'].cumsum()
hist_lv = ts_group.groupby('user_id')['cum_lv_incl'].shift(1).fillna(0)
ts_group['hist_long_view_rate'] = hist_lv / ts_group['hist_interaction_count']
ts_group.loc[ts_group['hist_interaction_count'] == 0, 'hist_long_view_rate'] = float('nan')

full = full.merge(ts_group[['user_id', 'time_ms', 'hist_long_view_rate']], on=['user_id', 'time_ms'], how='left')

user_feat = pd.read_csv(f"{DATA}/user_features_1k.csv")
video_feat = pd.read_csv(f"{DATA}/video_features_basic_1k.csv")

user_cols = ['user_id', 'is_lowactive_period', 'is_live_streamer', 'is_video_author',
             'follow_user_num', 'follow_user_num_range', 'fans_user_num', 'fans_user_num_range',
             'friend_user_num', 'friend_user_num_range', 'register_days', 'register_days_range'] + \
            [f'onehot_feat{i}' for i in range(18)]
user_feat_sub = user_feat[user_cols]

video_cols = ['video_id', 'video_type', 'upload_type', 'visible_status', 'server_width', 'server_height']
video_feat_sub = video_feat[video_cols]

full = full.merge(user_feat_sub, on='user_id', how='left')
full = full.merge(video_feat_sub, on='video_id', how='left')
full = full.dropna(subset=['visible_status', 'server_width', 'server_height', 'hist_long_view_rate'])

numeric_features = [
    'is_lowactive_period', 'is_live_streamer', 'is_video_author',
    'follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days',
    'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat5',
    'onehot_feat6', 'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11',
    'visible_status', 'server_width', 'server_height', 'duration_ms',
    'is_rand', 'tab',
]
cat_cols = ['video_type', 'upload_type', 'follow_user_num_range', 'fans_user_num_range',
            'friend_user_num_range', 'register_days_range',
            'onehot_feat4', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
            'onehot_feat15', 'onehot_feat16', 'onehot_feat17']

for c in cat_cols:
    full[c] = full[c].astype(str).replace('nan', '__MISSING__')

train_df = full[(full['source'] == 'standard') & (full['date'] <= 20220421)]
std_test_df = full[(full['source'] == 'standard') & (full['date'] >= 20220422)]
random_test_df = full[full['source'] == 'random']

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train_df[cat_cols])

def make_X(df):
    cat = pd.DataFrame(ohe.transform(df[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    num = df[numeric_features + ['hist_long_view_rate']].reset_index(drop=True).astype(float)
    return pd.concat([num, cat], axis=1)

X_train = make_X(train_df)
y_train = train_df['long_view'].reset_index(drop=True).astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
model.fit(X_train_scaled, y_train)

def evaluate(df, name):
    X = make_X(df)
    X_scaled = scaler.transform(X)
    y = df['long_view'].reset_index(drop=True).astype(int)
    proba = model.predict_proba(X_scaled)[:, 1]
    result = {
        'auc': roc_auc_score(y, proba),
        'log_loss': log_loss(y, proba),
        'brier': brier_score_loss(y, proba),
        'pr_auc': average_precision_score(y, proba),
    }
    print(f"\n=== {name}（n={len(df)}）===")
    print(f"AUC={result['auc']:.6f}")
    print(f"LogLoss={result['log_loss']:.6f}")
    print(f"Brier={result['brier']:.6f}")
    print(f"PR-AUC={result['pr_auc']:.6f}")
    return result

std_result = evaluate(std_test_df, "标准测试集（对照，应接近决策19的数字）")
random_result = evaluate(random_test_df, "随机曝光日志（泛化检验）")

print("\n=== 对比（随机曝光 - 标准测试集）===")
for metric in ['auc', 'log_loss', 'brier', 'pr_auc']:
    diff = random_result[metric] - std_result[metric]
    print(f"{metric}: {std_result[metric]:.6f} -> {random_result[metric]:.6f}, 差值={diff:+.6f}")

print("\n=== 正例比例核对（用实际评估时的同一份数据）===")
std_rate = std_test_df['long_view'].mean()
random_rate = random_test_df['long_view'].mean()
print(f"标准测试集（n={len(std_test_df)}）long_view 正例比例: {std_rate*100:.4f}%")
print(f"随机曝光日志（n={len(random_test_df)}）long_view 正例比例: {random_rate*100:.4f}%")
print(f"比例差异: {(random_rate-std_rate)*100:+.4f} 个百分点")
print(f"随机曝光的正例比例是标准测试集的 {random_rate/std_rate*100:.2f}%")
