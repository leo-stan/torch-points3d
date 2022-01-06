import laspy
import os
from joblib import Parallel, delayed
from welford import Welford
from tqdm import tqdm
import numpy as np


in_dir = "/media/ml2_ssd/orig-data"


def do_computefile(file, color_range):
    las = laspy.read(file)

    rgb = np.stack([las.red, las.green, las.blue], axis=1) / color_range
    w = Welford(rgb)
    return w


path = os.path.join(in_dir, "lux", "las")
w1s = Parallel(n_jobs=-1)(delayed(do_computefile)(os.path.join(path, file), 65535) for file in tqdm(os.listdir(path)))
path = os.path.join(in_dir, "basel", "las")
w2s = Parallel(n_jobs=-1)(delayed(do_computefile)(os.path.join(path, file), 255) for file in tqdm(os.listdir(path)))

w1s.extend(w2s)
w_tot = None
for w in w1s:
    if w_tot is None:
        w_tot = w
    else:
        w_tot.merge(w)

print("Overall stats:")
print("mean: ", w_tot.mean)
print("var_s: ", w_tot.var_s)
print("var_p: ", w_tot.var_p)
