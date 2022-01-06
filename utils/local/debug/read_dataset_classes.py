import laspy
import numpy as np
from joblib import Parallel, delayed
import os

base_dir = "/media/ml2_ssd/orig-data/basel/las"
def read_file(i, file):
    if i % 500 == 0:
        print("==================================== %d ================================" % i)
    try:
        x = laspy.read(os.path.join(base_dir, file), laz_backend=laspy.compression.LazBackend.Lazrs)
        unique_class = np.unique(x.classification)
        print(file, unique_class)
        return unique_class
    except:
        print("Error on file ", file)
        return np.array([])

unique_classes = Parallel(n_jobs=32)(
    delayed(read_file)(i, file)
    for i, file in enumerate(list(os.listdir(base_dir)))
)
print("overall unique")
y = np.unique(np.concatenate(unique_classes))
print(y)