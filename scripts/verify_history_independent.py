import pandas as pd

std = pd.read_csv("data_cleaned/standard_cleaned.csv")
train2 = pd.read_csv("data_cleaned/model2_features_train.csv")

sample_rows = train2.sample(5, random_state=42)

all_match = True
for idx, row in sample_rows.iterrows():
    uid = row['user_id']
    t = row['time_ms']

    prior = std[(std['user_id'] == uid) & (std['time_ms'] < t)]
    independent_count = len(prior)
    independent_long_view_count = prior['long_view'].sum()
    independent_rate = independent_long_view_count / independent_count if independent_count > 0 else float('nan')

    pipeline_count = row['hist_interaction_count']
    pipeline_long_view_count = row['hist_long_view_count']
    pipeline_rate = row['hist_long_view_rate']

    match = (pipeline_count == independent_count) and (pipeline_long_view_count == independent_long_view_count)
    if not match:
        all_match = False

    print(f"user_id={uid}, time_ms={t}")
    print(f"  流水线算出的: count={pipeline_count}, long_view_count={pipeline_long_view_count}, rate={pipeline_rate}")
    print(f"  独立重算的:   count={independent_count}, long_view_count={independent_long_view_count}, rate={independent_rate}")
    print(f"  一致: {match}\n")

print(f"全部样本是否完全一致: {all_match}")
