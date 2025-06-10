import glob
import open3d
import numpy as np

import logging

# 配置日志
logging.basicConfig(filename='example.log', level=logging.INFO)

path = "/mnt/d/实验存档/pmd_v10_2"

label_path = path + "/sample/gt/*/*"

predict_path = path + "/sample/pred/"


def calculate_fscore(gt: open3d.geometry.PointCloud, pr: open3d.geometry.PointCloud, th: float = 0.01):
    '''Calculates the F-score between two point clouds with the corresponding threshold value.'''
    # open3d.visualization.draw_geometries([gt, pr],
    #                                   window_name='Open3D',
    #                                   width=1920, height=1080,
    #                                   point_show_normal=False)
    d1 = pr.compute_point_cloud_distance(gt)
    d2 = gt.compute_point_cloud_distance(pr)

    if len(d1) and len(d2):
        recall = float(sum(d < th for d in d2)) / float(len(d2))
        precision = float(sum(d < th for d in d1)) / float(len(d1))

        if recall + precision > 0:
            fscore = 2 * recall * precision / (recall + precision)
        else:
            fscore = 0
    else:
        fscore = 0
        precision = 0
        recall = 0

    pr = np.asarray(pr.points)
    gt = np.asarray(gt.points)

    cd = (np.mean(d1) + np.mean(d2)) / 2

    return fscore, cd


def read_point_set(fp):
    pcd = open3d.io.read_point_cloud(fp)
    return pcd


labels = []
y_pred = []
revise = []

filename = []
filepath = []


class Map:

    def __init__(self) -> None:
        super().__init__()
        self.list = []

    def put(self, key, value):
        item = self.get(key)
        if item is not None:
            self.list.remove({
                key: item
            })
        self.list.append({
            key: value
        })

    def get(self, key):
        for item in self.list:
            for k in item:
                if k == key:
                    return item[key]
        return None


for f in glob.iglob(label_path):
    filepath.append(f)
    filename.append({
        'cat': f.split('/')[-2],
        'name': f.split('/')[-1]
    })
    labels.append({
        "pc": read_point_set(f),
        "cat": f.split('/')[-2]
    })

for f in filename:
    y_pred.append(read_point_set(predict_path + f['cat'] + '/' + f['name']))

# for f in glob.iglob(revise_path):
#     revise.append(read_point_set(f))


f_map = Map()
cd_map = Map()
cat_quantity = Map()
top = Map()
bottom = Map()

from tqdm import tqdm


class Score:
    name: str = None
    fs: float = None

    def __init__(self, name, fs):
        self.name = name
        self.fs = fs

    def get_fs(self):
        return self.fs

    def get_name(self):
        return self.name


def set_top_bottom(cat, fs, name):
    max_len = 30
    top_ = top.get(cat)
    bottom_ = bottom.get(cat)
    if top_ is None:
        top_ = []
    if bottom_ is None:
        bottom_ = []

    item = Score(name, fs)
    top_.append(item)
    bottom_.append(item)
    top_.sort(key=lambda i: i.get_fs(), reverse=True)
    bottom_.sort(key=lambda i: i.get_fs())
    if len(top_) > max_len:
        top_ = top_[:max_len]
    if len(bottom_) > max_len:
        bottom_ = bottom_[:max_len]
    top.put(cat, top_)
    bottom.put(cat, bottom_)


for i, label in tqdm(enumerate(labels), total=len(labels)):
    f_score = f_map.get(label["cat"])
    chamfer = cd_map.get(label["cat"])
    q = cat_quantity.get(label["cat"])
    if f_score is None:
        f_score = 0
        chamfer = 0
        q = 0
    q += 1
    pred = y_pred[i]
    fs, cd = calculate_fscore(pred, label["pc"])
    set_top_bottom(label["cat"], fs, filename[i]['name'])
    f_score += fs
    chamfer += cd
    f_map.put(label["cat"], f_score)
    cd_map.put(label["cat"], chamfer)
    cat_quantity.put(label["cat"], q)

print("--------------   f-score     --------------")
# print(f_map.list)

f_total = 0
for i, item in enumerate(f_map.list):
    for key in item:
        f_score = item[key] / cat_quantity.get(key)
        f_total += f_score
        print("--------   {}: {}     --------------".format(key, f_score))

print("--------   avg: {}     --------------".format(f_total / len(f_map.list)))

print("\n")

print("--------------   chamfer distance     --------------")
cd_total = 0
for i, item in enumerate(cd_map.list):
    for key in item:
        score = item[key] / cat_quantity.get(key)
        cd_total += score
        print("--------   {}:       {}   ".format(key, score))
print("--------   avg: {}     --------------\n\n".format(cd_total / len(f_map.list)))

logging.info("---------------------------  top -------------------------\n\n")
for i, item in enumerate(top.list):
    for key in item:
        score = item[key]
        logging.info("----   {}   ----".format(key))
        for fs in score:
            logging.info("-- {}:   {} --\n".format(fs.name, fs.get_fs()))
logging.info("----------------------------------------------------------")

print()
print()

logging.info("---------------------------  bottom -------------------------\n\n")
for i, item in enumerate(bottom.list):
    for key in item:
        score = item[key]
        logging.info("----   {}   ----".format(key))
        for fs in score:
            logging.info("-- {}:   {} --\n".format(fs.name, fs.get_fs()))
