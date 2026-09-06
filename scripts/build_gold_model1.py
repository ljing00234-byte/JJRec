import pandas as pd

DATA = "/Users/jj/Downloads/GOAI agent/JKRec/KuaiRand-1K/data"

train = pd.read_csv("data_cleaned/train_4_08_to_4_21.csv")
test = pd.read_csv("data_cleaned/test_4_22_to_5_08.csv")

user_feat = pd.read_csv(f"{DATA}/user_features_1k.csv")
video_feat = pd.read_csv(f"{DATA}/video_features_basic_1k.csv")

user_cols = ['user_id', 'is_lowactive_period', 'is_live_streamer', 'is_video_author',
             'follow_user_num', 'follow_user_num_range', 'fans_user_num', 'fans_user_num_range',
             'friend_user_num', 'friend_user_num_range', 'register_days', 'register_days_range'] + \
            [f'onehot_feat{i}' for i in range(18)]
user_feat_sub = user_feat[user_cols]

video_cols = ['video_id', 'video_type', 'upload_type', 'visible_status', 'server_width', 'server_height']
video_feat_sub = video_feat[video_cols]


def build_gold(df, name):
    before = len(df)
    merged = df.merge(user_feat_sub, on='user_id', how='left')
    merged = merged.merge(video_feat_sub, on='video_id', how='left')

    dropped = merged[['visible_status', 'server_width', 'server_height']].isnull().any(axis=1).sum()
    merged = merged.dropna(subset=['visible_status', 'server_width', 'server_height'])
    print(f"{name}: join前行数={before}, join后行数={len(merged) + dropped}, "
          f"因 visible_status/server_width/server_height 缺失剔除={dropped}, 最终行数={len(merged)}")

    return merged


gold_train = build_gold(train, "train")
gold_test = build_gold(test, "test")

cat_cols = ['video_type', 'upload_type', 'follow_user_num_range', 'fans_user_num_range',
            'friend_user_num_range', 'register_days_range',
            'onehot_feat4', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
            'onehot_feat15', 'onehot_feat16', 'onehot_feat17']

combined = pd.concat([gold_train.assign(_split='train'), gold_test.assign(_split='test')], ignore_index=True)
combined = pd.get_dummies(combined, columns=cat_cols, dummy_na=True)

gold_train_final = combined[combined['_split'] == 'train'].drop(columns=['_split'])
gold_test_final = combined[combined['_split'] == 'test'].drop(columns=['_split'])

print(f"\n最终 Gold 表列数: {gold_train_final.shape[1]}")
print(f"train 形状: {gold_train_final.shape}")
print(f"test 形状: {gold_test_final.shape}")
print(f"train 缺失值总数: {gold_train_final.isnull().sum().sum()}")
print(f"test 缺失值总数: {gold_test_final.isnull().sum().sum()}")

gold_train_final.to_csv("data_cleaned/gold_model1_train.csv", index=False)
gold_test_final.to_csv("data_cleaned/gold_model1_test.csv", index=False)
print("\nGold层 model1 训练/测试表已保存")
