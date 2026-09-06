import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

DATA = os.environ.get("KUAIRAND_DATA_DIR")
if not DATA:
    raise RuntimeError("请设置环境变量 KUAIRAND_DATA_DIR，指向本地 KuaiRand-1K/data 目录")

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
print(f"train行数={len(train_df)}, test行数={len(test_df)}")

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

HYPERPARAMS = dict(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)

def train_and_eval(hist_features_to_use):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_df[cat_cols])

    train_cat = pd.DataFrame(ohe.transform(train_df[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    test_cat = pd.DataFrame(ohe.transform(test_df[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))

    X_train = pd.concat([
        train_df[numeric_features].reset_index(drop=True).astype(float),
        train_df[hist_features_to_use].reset_index(drop=True).astype(float),
        train_cat
    ], axis=1)
    X_test = pd.concat([
        test_df[numeric_features].reset_index(drop=True).astype(float),
        test_df[hist_features_to_use].reset_index(drop=True).astype(float),
        test_cat
    ], axis=1)

    y_train = train_df['long_view'].reset_index(drop=True).astype(int)
    y_test = test_df['long_view'].reset_index(drop=True).astype(int)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(**HYPERPARAMS)
    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    return dict(
        auc=roc_auc_score(y_test, proba),
        log_loss=log_loss(y_test, proba),
        brier=brier_score_loss(y_test, proba),
        pr_auc=average_precision_score(y_test, proba),
        n_features=X_train.shape[1],
        features=list(X_train.columns),
    )

print("\n=== model2（只有历史长播率）===")
r_base = train_and_eval(['hist_long_view_rate'])
print(f"特征数: {r_base['n_features']}")
print(f"AUC={r_base['auc']:.6f}, LogLoss={r_base['log_loss']:.6f}, "
      f"Brier={r_base['brier']:.6f}, PR-AUC={r_base['pr_auc']:.6f}")

print("\n=== model2+（全部5个历史特征）===")
r_all = train_and_eval(hist_cols)
print(f"特征数: {r_all['n_features']}（应比上面多4）")
print(f"多出的历史特征: {[c for c in hist_cols if c != 'hist_long_view_rate']}")
print(f"AUC={r_all['auc']:.6f}, LogLoss={r_all['log_loss']:.6f}, "
      f"Brier={r_all['brier']:.6f}, PR-AUC={r_all['pr_auc']:.6f}")

print("\n=== 对比（model2+ - model2）===")
for metric in ['auc', 'log_loss', 'brier', 'pr_auc']:
    diff = r_all[metric] - r_base[metric]
    print(f"{metric}: {r_base[metric]:.6f} -> {r_all[metric]:.6f}, 差值={diff:+.6f}")
