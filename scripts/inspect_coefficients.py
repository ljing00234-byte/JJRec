import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder

DATA = "/Users/jj/Downloads/GOAI agent/JKRec/KuaiRand-1K/data"

m2_train = pd.read_csv("data_cleaned/model2_features_train.csv")
m2_test = pd.read_csv("data_cleaned/model2_features_test.csv")

user_feat = pd.read_csv(f"{DATA}/user_features_1k.csv")
video_feat = pd.read_csv(f"{DATA}/video_features_basic_1k.csv")

user_cols = ['user_id', 'is_lowactive_period', 'is_live_streamer', 'is_video_author',
             'follow_user_num', 'follow_user_num_range', 'fans_user_num', 'fans_user_num_range',
             'friend_user_num', 'friend_user_num_range', 'register_days', 'register_days_range'] + \
            [f'onehot_feat{i}' for i in range(18)]
user_feat_sub = user_feat[user_cols]

video_cols = ['video_id', 'video_type', 'upload_type', 'visible_status', 'server_width', 'server_height']
video_feat_sub = video_feat[video_cols]

hist_cols = ['hist_long_view_rate', 'hist_is_like_rate', 'hist_is_comment_rate',
             'hist_is_forward_rate', 'hist_is_profile_enter_rate']

def build(df):
    df = df.merge(user_feat_sub, on='user_id', how='left')
    df = df.merge(video_feat_sub, on='video_id', how='left')
    df = df.dropna(subset=['visible_status', 'server_width', 'server_height'] + hist_cols)
    return df

train_df = build(m2_train)
test_df = build(m2_test)

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
    train_df[c] = train_df[c].astype(str).replace('nan', '__MISSING__')
    test_df[c] = test_df[c].astype(str).replace('nan', '__MISSING__')

ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
ohe.fit(train_df[cat_cols])
train_cat = pd.DataFrame(ohe.transform(train_df[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))

X_train = pd.concat([
    train_df[numeric_features].reset_index(drop=True).astype(float),
    train_df[hist_cols].reset_index(drop=True).astype(float),
    train_cat
], axis=1)
y_train = train_df['long_view'].reset_index(drop=True).astype(int)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

model = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
model.fit(X_train_scaled, y_train)

coef_map = dict(zip(X_train.columns, model.coef_[0]))

print("=== 5个历史特征各自的系数（标准化后，可直接比较大小）===")
for c in hist_cols:
    print(f"{c}: {coef_map[c]:.6f}")

base_abs = abs(coef_map['hist_long_view_rate'])
print(f"\n以历史长播率系数的绝对值（{base_abs:.6f}）为参照：")
for c in hist_cols[1:]:
    ratio = abs(coef_map[c]) / base_abs
    print(f"{c}: 绝对值是长播率系数的 {ratio:.4f} 倍")
