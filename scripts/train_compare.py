import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss, average_precision_score

static_features = [
    'is_lowactive_period', 'is_live_streamer', 'is_video_author',
    'follow_user_num', 'fans_user_num', 'friend_user_num', 'register_days',
    'onehot_feat0', 'onehot_feat1', 'onehot_feat2', 'onehot_feat3', 'onehot_feat5',
    'onehot_feat6', 'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 'onehot_feat11',
    'visible_status', 'server_width', 'server_height', 'duration_ms',
    'is_rand', 'tab',
]

onehot_prefixes = ['follow_user_num_range', 'fans_user_num_range', 'friend_user_num_range',
                   'register_days_range', 'onehot_feat4', 'onehot_feat12', 'onehot_feat13',
                   'onehot_feat14', 'onehot_feat15', 'onehot_feat16', 'onehot_feat17',
                   'video_type', 'upload_type']


def get_onehot_cols(df, prefixes):
    cols = []
    for p in prefixes:
        cols += [c for c in df.columns if c.startswith(p + '_')]
    return cols


train1 = pd.read_csv("data_cleaned/gold_model1_train_aligned.csv")
test1 = pd.read_csv("data_cleaned/gold_model1_test_aligned.csv")
train2 = pd.read_csv("data_cleaned/gold_model2_train.csv")
test2 = pd.read_csv("data_cleaned/gold_model2_test.csv")

onehot_cols1 = get_onehot_cols(train1, onehot_prefixes)
onehot_cols2 = get_onehot_cols(train2, onehot_prefixes)

X1_cols = static_features + onehot_cols1
X2_cols = static_features + onehot_cols2 + ['hist_long_view_rate']

print(f"model1 特征数: {len(X1_cols)}")
print(f"model1 特征列表: {X1_cols}")
print(f"\nmodel2 特征数: {len(X2_cols)}")
print(f"model2 特征列表（多出的部分）: {[c for c in X2_cols if c not in X1_cols]}")


def prepare(df, cols):
    X = df[cols].astype(float)
    y = df['long_view'].astype(int)
    return X, y


X1_train, y1_train = prepare(train1, X1_cols)
X1_test, y1_test = prepare(test1, X1_cols)
X2_train, y2_train = prepare(train2, X2_cols)
X2_test, y2_test = prepare(test2, X2_cols)

HYPERPARAMS = dict(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)


def train_and_eval(X_train, y_train, X_test, y_test, name):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(**HYPERPARAMS)
    model.fit(X_train_scaled, y_train)

    proba = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, proba)
    ll = log_loss(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    print(f"\n=== {name} ===")
    print(f"AUC: {auc:.6f}")
    print(f"Log Loss: {ll:.6f}")
    print(f"Brier Score: {brier:.6f}")
    print(f"PR-AUC: {pr_auc:.6f}")

    return dict(auc=auc, log_loss=ll, brier=brier, pr_auc=pr_auc)


result1 = train_and_eval(X1_train, y1_train, X1_test, y1_test, "model1（不含历史）")
result2 = train_and_eval(X2_train, y2_train, X2_test, y2_test, "model2（含历史长播率）")

print("\n=== 对比（model2 - model1）===")
for metric in ['auc', 'log_loss', 'brier', 'pr_auc']:
    diff = result2[metric] - result1[metric]
    print(f"{metric}: model1={result1[metric]:.6f}, model2={result2[metric]:.6f}, 差值={diff:+.6f}")
