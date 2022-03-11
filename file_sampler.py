import random

import pandas as pd
import os
import numpy as np


def sample_files(source_name: str, target_folder: str = "samples", samples=1, coloumns_to_drop=[],
                 inlier_classes=[], max_data: int = 950, inlier_fraction=0.95, whitespace_delim: bool = False):
    file = pd \
        .read_csv(os.path.join(os.path.dirname(__file__), "datasets", source_name), delim_whitespace=whitespace_delim) \
        .drop(columns=coloumns_to_drop) \

    file.insert(len(file.columns)-1, 'class', file.pop('class'))
    file['class'] = file['class'].map(lambda x: 1 if x in inlier_classes else -1)

    obj = file.sort_values(by='class').values
    labels, counts = np.unique(obj[:, -1], return_counts=True)

    inlier = []
    outlier = []
    current_start = 0
    for i in range(len(counts)):
        if labels[i] == 1:
            inlier.extend(list(range(current_start, current_start + counts[i])))
        else:
            outlier.extend(list(range(current_start, current_start + counts[i])))

        current_start += counts[i]

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for i in range(samples):
        samples_indices = random.sample(inlier, int(min(len(inlier), int(max_data * inlier_fraction))))
        samples_indices.extend(
            random.sample(outlier, int(len(samples_indices) / inlier_fraction * (1 - inlier_fraction))))

        samples = obj[samples_indices]

        filename = source_name.split(".")[0] + "_sample_" + str(i) + "_data-points_" + str(
            len(samples)) + "_inlier-classes_" + str(inlier_classes) + "_inlier-fraction_" + str(
            inlier_fraction) + ".csv"

        pd.DataFrame(samples).to_csv(os.path.join(target_folder, filename), index=False, header=False)


# list meanings: str-file name, bool - whitespace_delim, list - drop columns, list - inlier classes
datasets = [
    ["glass.data", False, ["index"], [1]],
    ["ecoli.data", True, ["sample_machine"], ["cp"]],
    ["lymphography.data", False, [], [2]],
]

for dataset in datasets:
    sample_files(dataset[0], target_folder=os.path.join("samples", dataset[0].split(".")[0]),
                 whitespace_delim=dataset[1], samples=5,
                 coloumns_to_drop=dataset[2], inlier_classes=dataset[3])
