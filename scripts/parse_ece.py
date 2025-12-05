import json

try:
    with open('/home/mohanganesh/retail-shelf-detection/results/ssod_baselines/consistent_teacher/summary.json') as f:
        print("Real CT mAP:", json.load(f)["mean_map"])
except: pass
