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
full = pd.concat([m2_train, m2_test], ignore_index=True)

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

print(f"完整表行数: {len(full)}")

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

# 分类字段缺失值先统一标记为字符串"__MISSING__"，作为独立类别处理
for c in cat_cols:
    full[c] = full[c].astype(str).fillna('__MISSING__')
    full.loc[full[c] == 'nan', c] = '__MISSING__'

HYPERPARAMS = dict(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)


def train_and_eval(train_df, test_df, use_history):
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    ohe.fit(train_df[cat_cols])

    train_cat = ohe.transform(train_df[cat_cols])
    test_cat = ohe.transform(test_df[cat_cols])

    X_train = pd.concat([
        train_df[numeric_features].reset_index(drop=True).astype(float),
        pd.DataFrame(train_cat, columns=ohe.get_feature_names_out(cat_cols))
    ], axis=1)
    X_test = pd.concat([
        test_df[numeric_features].reset_index(drop=True).astype(float),
        pd.DataFrame(test_cat, columns=ohe.get_feature_names_out(cat_cols))
    ], axis=1)

    if use_history:
        X_train['hist_long_view_rate'] = train_df['hist_long_view_rate'].reset_index(drop=True)
        X_test['hist_long_view_rate'] = test_df['hist_long_view_rate'].reset_index(drop=True)

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
    )


split_dates = [20220415, 20220421, 20220429]
all_results = []

for split_date in split_dates:
    train_df = full[full['date'] <= split_date]
    test_df = full[full['date'] > split_date]
    print(f"\n{'='*20} 切分日期: {split_date} {'='*20}")
    print(f"train行数={len(train_df)}, test行数={len(test_df)}")

    r1 = train_and_eval(train_df, test_df, use_history=False)
    r2 = train_and_eval(train_df, test_df, use_history=True)

    print(f"model1(特征数={r1['n_features']}): AUC={r1['auc']:.4f}, LogLoss={r1['log_loss']:.4f}, "
          f"Brier={r1['brier']:.4f}, PR-AUC={r1['pr_auc']:.4f}")
    print(f"model2(特征数={r2['n_features']}): AUC={r2['auc']:.4f}, LogLoss={r2['log_loss']:.4f}, "
          f"Brier={r2['brier']:.4f}, PR-AUC={r2['pr_auc']:.4f}")
    print(f"差值: AUC={r2['auc']-r1['auc']:+.4f}, LogLoss={r2['log_loss']-r1['log_loss']:+.4f}, "
          f"Brier={r2['brier']-r1['brier']:+.4f}, PR-AUC={r2['pr_auc']-r1['pr_auc']:+.4f}")

    all_results.append((split_date, r1, r2))

print(f"\n{'='*20} 汇总：三次切分的差值方向是否一致 {'='*20}")
for metric in ['auc', 'log_loss', 'brier', 'pr_auc']:
    diffs = [r2[metric] - r1[metric] for _, r1, r2 in all_results]
    print(f"{metric}: {[f'{d:+.4f}' for d in diffs]}")
