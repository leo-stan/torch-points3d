import laspy
import os
from joblib import Parallel, delayed
from welford import Welford
from tqdm import tqdm
import numpy as np


in_dir = "/media/ml2_ssd/orig-data"


def do_computefile(file):
    try:
        las = laspy.read(file)

        uni = np.unique(las.classification, return_counts=True)
        return uni
    except:
        print("error on file %s" % file)
        return (np.array([]), np.array([]))


# path = os.path.join(in_dir, "montreal", "las")
# unis = Parallel(n_jobs=-1)(delayed(do_computefile)(os.path.join(path, file)) for file in tqdm(os.listdir(path)))
path = os.path.join(in_dir, "tyrol", "las")
unis = Parallel(n_jobs=-1)(delayed(do_computefile)(os.path.join(path, file)) for file in tqdm(os.listdir(path)))

uni_cts = {}
for class_codes, counts in unis:
    for class_code, count in zip(class_codes, counts):
        if not class_code in uni_cts:
            uni_cts[class_code] = 0
        uni_cts[class_code] += count

print("Overall counts:")
print(uni_cts)
