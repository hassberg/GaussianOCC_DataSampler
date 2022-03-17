import pandas as pd
import os
import numpy as np

from sklearn.model_selection import train_test_split

train_fraction = 0.7
test_fraction = 0.15
eval_fraction = 0.15


def sample_files(source_name: str, target_folder: str = "samples", samples=1, coloumns_to_drop=[],
                 inlier_classes=[], max_data: int = 1000, inlier_fraction=0.95, whitespace_delim: bool = False):
    file = pd \
        .read_csv(os.path.join(os.path.dirname(__file__), "datasets", source_name), delim_whitespace=whitespace_delim,
                  index_col=False) \
        .drop(columns=coloumns_to_drop)

    file.insert(len(file.columns) - 1, 'class', file.pop('class'))
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
        inlier_count = min(len(inlier), max_data * inlier_fraction * 1 / train_fraction)

        train_inlier, remaining = train_test_split(inlier, train_size=int(inlier_count * train_fraction))
        test_inlier, eval_inlier = train_test_split(remaining, train_size=int(inlier_count * test_fraction))

        train_outlier, remaining = train_test_split(outlier, train_size=int(
            len(train_inlier) / inlier_fraction * (1 - inlier_fraction)))
        test_outlier, eval_outlier = train_test_split(outlier, train_size=np.maximum(3, int(len(
            test_inlier) / inlier_fraction * (1 - inlier_fraction))))  # TODO max 3 ok?

        samples_train_indices = train_inlier + train_outlier
        samples_test_indices = test_inlier + test_outlier
        samples_eval_indices = eval_inlier + eval_outlier

        samples_train = obj[samples_train_indices]
        samples_test = obj[samples_test_indices]
        samples_eval = obj[samples_eval_indices]

        classes = ','.join(str(e) for e in inlier_classes)
        folder_name = source_name.split(".")[0] + "-" + classes + "_sample_" + str(i) + "_data-points_" + str(
            len(samples_train)) + "_inlier-fraction_" + str(inlier_fraction) + "_datasample"

        folder = os.path.join(target_folder, folder_name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        pd.DataFrame(samples_train).to_csv(os.path.join(folder, "train.csv"), index=False, header=False)
        pd.DataFrame(samples_test).to_csv(os.path.join(folder, "test.csv"), index=False, header=False)
        pd.DataFrame(samples_eval).to_csv(os.path.join(folder, "eval.csv"), index=False, header=False)


# list meanings: str-file name, bool - whitespace_delim, list - drop columns, list - inlier classes
datasets = [
    # ["breast-cancer-wisconsin.data", False, ["code"], [2]],
    # ["breast-cancer-wisconsin.data", False, ["code"], [4]],
    ["ecoli.data", True, ["sample_machine"], ["cp"]],
    # ["ecoli.data", True, ["sample_machine"], ["im"]],
    # ["glass.data", False, ["index"], [1]],
    # ["glass.data", False, ["index"], [2]],
    # ["ionosphere.data", False, [], ["g"]],
    # ["ionosphere.data", False, [], ["b"]],
    # ["lymphography.data", False, [], [2]],
    # ["lymphography.data", False, [], [3]],
    ["page-blocks.data", True, [], [1]],
    ["page-blocks.data", True, [], [2]],
    ["waveform.data", False, [], [0]],
    # ["waveform.data", False, [], [1]],
    # ["waveform.data", False, [], [2]],
    # ["wdbc.data", False, ["ID"], ["B"]],
    # ["wdbc.data", False, ["ID"], ["M"]],
    # ["wpbc.data", False, ["ID"], ["N"]],
    # ["wpbc.data", False, ["ID"], ["R"]],
    # ["yeast.data", True, ["sample_machine"], ["CYT"]],
    # ["yeast.data", True, ["sample_machine"], ["NUC"]],
    # ["yeast.data", True, ["sample_machine"], ["ME3"]],
    # ["yeast.data", True, ["sample_machine"], ["MIT"]],
]

for dataset in datasets:
    sample_files(dataset[0], target_folder=os.path.join("samples", dataset[0].split(".")[0]),
                 whitespace_delim=dataset[1], samples=1,
                 coloumns_to_drop=dataset[2], inlier_classes=dataset[3])
