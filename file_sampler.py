import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

train_fraction = 0.7
eval_fraction = 0.3


def create_sample(inlier_indices, outlier_indices, data_obj, output_path, sample_run, creating_test: bool = False, inlier_fraction=0.95, max_data: int = 500):
    inlier_count = min(len(inlier_indices), max_data * inlier_fraction * 1 / train_fraction)

    train_inlier, test_inlier = train_test_split(inlier_indices, train_size=int(inlier_count * train_fraction))

    # TODO relevant? if not eval_creation_mode:
    train_outlier, remaining = train_test_split(outlier_indices, train_size=int(len(train_inlier) / inlier_fraction * (1 - inlier_fraction)))
    if creating_test:
        test_outlier, _ = train_test_split(remaining, train_size=np.maximum(3, int(len(test_inlier) / inlier_fraction * (1 - inlier_fraction))))
    else:
        test_outlier, _ = train_test_split(remaining, train_size=np.minimum(len(remaining) - 1, 400))

    samples_train_indices = train_inlier + train_outlier
    samples_test_indices = test_inlier + test_outlier

    samples_train = data_obj[samples_train_indices]
    samples_test = data_obj[samples_test_indices]

    final_folder = os.path.join(output_path, sample_run)
    if not os.path.exists(final_folder):
        os.makedirs(final_folder)

    pd.DataFrame(samples_train).to_csv(os.path.join(final_folder, "train.csv"), index=False, header=False)
    pd.DataFrame(samples_test).to_csv(os.path.join(final_folder, "test.csv"), index=False, header=False)


def sample_files(source_name: str, target_folder: str, sampling_strategy: str, samples=5, coloumns_to_drop=[],
                 inlier_classes=[], whitespace_delim: bool = False):
    file = pd \
        .read_csv(os.path.join(os.path.dirname(__file__), "datasets", source_name), delim_whitespace=whitespace_delim, index_col=False) \
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

    classes = ','.join(str(e) for e in inlier_classes)
    folder_name = source_name.split(".")[0] + "-" + classes
    path = os.path.join(target_folder, folder_name)
    if not os.path.exists(path):
        os.makedirs(path)

    if sampling_strategy == "continuous":
        gt_path = os.path.join(path, "ground_truth")
        if not os.path.exists(gt_path):
            os.makedirs(gt_path)
        pd.DataFrame(obj).to_csv(os.path.join(gt_path, "ground_truth.csv"), index=False, header=False)

    ## sample train
    create_sample(inlier_indices=inlier, outlier_indices=outlier, sample_run="test", creating_test=True, data_obj=obj, output_path=path)
    for i in range(samples):
        ## sample tests
        create_sample(inlier_indices=inlier, outlier_indices=outlier, sample_run="sample_" + str(i), data_obj=obj, output_path=path)


# list meanings: str-file name, bool - whitespace_delim, list - drop columns, list - inlier classes - sampling_type
datasets = [
    # ["breast-cancer-wisconsin.data", False, ["code"], [2], "discrete", ],
    # ["breast-cancer-wisconsin.data", False, ["code"], [4], "discrete", ],
    ["ecoli.data", True, ["sample_machine"], ["cp"], "discrete", ],
    # ["ecoli.data", True, ["sample_machine"], ["im"], "discrete", ],
    # ["glass.data", False, ["index"], [1, 2, 3, 4], "discrete", ],
    # ["glass.data", False, ["index"], [2], "discrete", ],
    # ["ionosphere.data", False, [], ["g"], "discrete", ],
    # ["ionosphere.data", False, [], ["b"], "discrete", ],
    # ["lymphography.data", False, [], [2], "discrete", ],
    # ["lymphography.data", False, [], [3], "discrete", ],
    # ["page-blocks.data", True, [], [1], "discrete", ],
    # ["page-blocks.data", True, [], [2], "discrete", ],
    # ["svmguide1.csv", False, [], [1], "discrete", ],
    ["svmguide1.csv", False, [], [1], "continuous", ],
    # ["svmguide1.csv", False, [], [1], "discrete", ],
    # ["waveform.data", False, [], [0], "discrete", ],
    # ["waveform.data", False, [], [1], "discrete", ],
    # ["waveform.data", False, [], [2], "discrete", ],
    # ["wdbc.data", False, ["ID"], ["B"], "discrete", ],
    # ["wdbc.data", False, ["ID"], ["M"], "discrete", ],
    # ["wpbc.data", False, ["ID"], ["N"], "discrete", ],
    # ["wpbc.data", False, ["ID"], ["R"], "discrete", ],
    # ["yeast.data", True, ["sample_machine"], ["CYT"], "discrete", ],
    # ["yeast.data", True, ["sample_machine"], ["NUC"], "discrete", ],
    # ["yeast.data", True, ["sample_machine"], ["ME3"], "discrete", ],
    # ["yeast.data", True, ["sample_machine"], ["MIT"], "discrete", ],
]

for dataset in datasets:
    sample_files(dataset[0], target_folder="samples/" + dataset[4], sampling_strategy=dataset[4],
                 whitespace_delim=dataset[1], samples=10,
                 coloumns_to_drop=dataset[2], inlier_classes=dataset[3])
