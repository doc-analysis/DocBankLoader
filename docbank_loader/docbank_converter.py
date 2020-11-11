import os
import random
import shutil
from tqdm import tqdm
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw


from .docbank_loader import DocBankLoader, TokenInfo
from .loader import Loader

random.seed(42)
np.random.seed(42)

class Bbox:
    def __init__(self, bbox, structure, pagesize):
        self.bbox = bbox
        self.structure = structure
        self.pagesize = pagesize
    
    def __str__(self):
        return '\t'.join(list(map(str, self.bbox)) + [self.structure])

class NormalizedBbox:
    def __init__(self, bbox, structure):
        self.bbox = bbox
        self.structure = structure
    
    def __str__(self):
        return '\t'.join(list(map(str, self.bbox)) + [self.structure])

    def denormalize(self, pagesize):
        width, height = pagesize
        x0, y0, x1, y1 = self.bbox
        x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
            y1 * height / 1000)

        return Bbox([x0, y0, x1, y1], self.structure, pagesize)

class CVStructure:
    def __init__(self, infos, structure):
        self.infos = infos
        self.structure = structure

    def to_bbox(self):
        lefts = [t.bbox[0] for t in self.infos]
        tops = [t.bbox[1] for t in self.infos]
        rights = [t.bbox[2] for t in self.infos]
        bottoms = [t.bbox[3] for t in self.infos]

        return NormalizedBbox((min(lefts), min(tops), max(rights), max(bottoms)), self.structure)

    @classmethod
    def from_example(cls, example):
        infos = [t for t in TokenInfo.from_example(example)]
        flags = np.zeros(len(infos), dtype=int)

        struct_dict = {}
        for info_id, info in enumerate(infos):
            struct = info.structure
            if struct in struct_dict:
                struct_dict[struct].append((info, info_id))
            else:
                struct_dict[struct] = [(info, info_id)]

        cv_structures = []
        for struct in struct_dict.keys():
            struct_list = [t for t in struct_dict[struct] if flags[t[1]] == 0]
            while struct_list:
                this_info, this_info_id = random.choice(struct_list)

                cv_structure_infos = []

                queue = [this_info]
                flags[this_info_id] = 1

                while queue:
                    this_info = queue[0]
                    for info, info_id in struct_list:
                        if flags[info_id] == 0 and TokenInfo.is_neighbor(this_info, info):
                            queue.append(info)
                            flags[info_id] = 1
                    cv_structure_infos.append(queue.pop(0))

                cv_structures.append(CVStructure(cv_structure_infos, struct))
                struct_list = [
                    t for t in struct_dict[struct] if flags[t[1]] == 0]

        return cv_structures


class CVExample:
    def __init__(self, example, cv_structures):
        self.filepath = example.filepath
        self.pagesize = example.pagesize
        self.cv_structures = cv_structures
        self._bboxes = None

    @property
    def bboxes(self):
        if not self._bboxes:            
            bboxes = []
            for cv_structure in self.cv_structures:
                bboxes.append(cv_structure.to_bbox())
            self._bboxes = bboxes
        return self._bboxes


    def plot(self):    
        width, height = self.pagesize
        im = np.zeros(list(self.pagesize) + [3], dtype=np.uint8)

        for cv_structure in self.cv_structures:
            color = np.random.randint(256, size=3)
            for info in cv_structure.infos:

                x0, y0, x1, y1 = info.bbox
                x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                    y1 * height / 1000)

                for x in range(x0, x1):
                    for y in range(y0, y1):
                        im[x, y] = color

        im = np.swapaxes(im, 0, 1)
        im = Image.fromarray(im, mode='RGB')

        return im

    def plot_bbox(self):            
        width, height = self.pagesize
        im = Image.open(self.filepath)
        drawer = ImageDraw.Draw(im)

        for bbox in self.bboxes:
            color = tuple(np.random.randint(256, size=3))

            x0, y0, x1, y1 = bbox.bbox
            x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                y1 * height / 1000)

            drawer.text([max(x0-drawer.textsize(bbox.structure)[0], 0), y0], bbox.structure, fill=color)
            drawer.rectangle([x0, y0, x1, y1], outline=color)

        return im

    def print_bbox(self):
        bboxes = []
        for nor_bbox in self.bboxes:
            bbox = nor_bbox.denormalize(self.pagesize)
            bboxes.append(bbox)

        return '\n'.join(map(str, bboxes))



class DocBankConverter(Loader):
    def __init__(self, docbank):
        self.docbank = docbank
        self.basename_list = self.docbank.basename_list
        self.txt_dir = self.docbank.txt_dir
        self.img_dir = self.docbank.img_dir
        

    @classmethod
    def count_tolerance(cls, example):
        infos = TokenInfo.from_example(example)

        struct_dict = {}
        for info_id, info in enumerate(infos):
            struct = info.structure
            if struct in struct_dict:
                struct_dict[struct].append((info, info_id))
            else:
                struct_dict[struct] = [(info, info_id)]

        x_toler = []
        y_toler = []
        for struct in struct_dict.keys():
            struct_list = struct_dict[struct]

            if not struct_list:
                continue

            for i in range(len(struct_list) - 1):
                info0 = struct_list[i][0]
                info1 = struct_list[i + 1][0]

                x_toler.append(abs(info1.bbox[0] - info0.bbox[2]))
                y_toler.append(abs(info1.bbox[1] - info0.bbox[3]))

        x_tol_counter = Counter(x_toler)
        y_tol_counter = Counter(y_toler)

        return x_tol_counter, y_tol_counter        

    def sample_n(self, n):
        examples = self.docbank.sample_n(n)
        cv_examples = []
        for example in tqdm(examples, desc='Converting:'):
            cv_structures = CVStructure.from_example(example)
            cv_examples.append(CVExample(example, cv_structures))
        return cv_examples

    def read_all(self):
        examples = self.docbank.read_all()
        cv_examples = []
        for example in tqdm(examples, desc='Converting:'):
            cv_structures = CVStructure.from_example(example)
            cv_examples.append(CVExample(example, cv_structures))
        return cv_examples

    def get_by_filename(self, filename):
        example = self.docbank.get_by_filename(filename)
        cv_structures = CVStructure.from_example(example)
        segment_example = CVExample(example, cv_structures)
        return segment_example

