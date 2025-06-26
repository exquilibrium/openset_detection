import ast
from collections import Counter

log_fn = '/home/chen/openset_detection/log_associate_data.txt'

# Initialize sum list with 5 zeros
sums = [0] * 5

with open(log_fn, "r") as f:
    for line in f:
        # Convert string line to list using ast.literal_eval for safety
        values = ast.literal_eval(line.strip())
        for i in range(5):
            sums[i] += values[i]

print("Sums per index:", sums, end='\n\n')

# Count occurrences of values[i] for each index value
dets = [{} for _ in range(5)]

with open(log_fn, "r") as f:
    for line in f:
        values = ast.literal_eval(line.strip())
        for i in range(5):
            val = values[i]
            if val in dets[i]:
                dets[i][val] += 1
            else:
                dets[i][val] = 1

type_map = [
    'Correct classification',
    'Misclassification',
    'Ignored',
    'Unknown classification',
    'Background'
]
for i in range(5):
    sum = 0
    print(f"{type_map[i]}:")
    sorted_dict = dict(sorted(dets[i].items()))
    for k, v in sorted_dict.items():
        print("    ", k, v)
        sum += v
    print("Sum of vals:", sum)
