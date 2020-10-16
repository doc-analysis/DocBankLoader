import os
import random
import shutil
from tqdm import tqdm
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw


from docbank_reader import DocBankReader, TokenInfo
from reader import Reader

# random.seed(42)
np.random.seed(42)

class Bbox:
    def __init__(self, bbox, structure):
        self.bbox = bbox
        self.structure = structure

class Segmentation:
    def __init__(self, infos, structure):
        self.infos = infos
        self.structure = structure

    def to_bbox(self):
        lefts = [t.bbox[0] for t in self.infos]
        tops = [t.bbox[1] for t in self.infos]
        rights = [t.bbox[2] for t in self.infos]
        bottoms = [t.bbox[3] for t in self.infos]

        return Bbox((min(lefts), min(tops), max(rights), max(bottoms)), self.structure)

    @classmethod
    def from_example(cls, example):
        infos = TokenInfo.from_example(example)
        flags = np.zeros(len(infos), dtype=int)

        struct_dict = {}
        for info_id, info in enumerate(infos):
            struct = info.structure
            if struct in struct_dict:
                struct_dict[struct].append((info, info_id))
            else:
                struct_dict[struct] = [(info, info_id)]

        segments = []
        for struct in struct_dict.keys():
            struct_list = [t for t in struct_dict[struct] if flags[t[1]] == 0]
            while struct_list:
                this_info, this_info_id = random.choice(struct_list)

                segment_infos = []

                queue = [this_info]
                flags[this_info_id] = 1

                while queue:
                    this_info = queue[0]
                    for info, info_id in struct_list:
                        if flags[info_id] == 0 and TokenInfo.is_neighbor(this_info, info):
                            queue.append(info)
                            flags[info_id] = 1
                    segment_infos.append(queue.pop(0))

                segments.append(Segmentation(segment_infos, struct))
                struct_list = [
                    t for t in struct_dict[struct] if flags[t[1]] == 0]

        return segments


class SegmentationExample:
    def __init__(self, example, segments):
        self.filepath = example.filepath
        self.pagesize = example.pagesize
        self.segments = segments
        self._bboxes = None

    @property
    def bboxes(self):
        if not self._bboxes:            
            bboxes = []
            for segment in self.segments:
                bboxes.append(segment.to_bbox())
            self._bboxes = bboxes
        return self._bboxes


    def plot(self):    
        width, height = self.pagesize
        im = np.zeros(list(self.pagesize) + [3], dtype=np.uint8)

        for segment in self.segments:
            color = np.random.randint(256, size=3)
            for info in segment.infos:

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

            drawer.rectangle([x0, y0, x1, y1], outline=color)

        return im


class DocBankSegmentationReader(Reader):
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
        segment_examples = []
        for example in tqdm(examples, desc='Converting:'):
            segments = Segmentation.from_example(example)
            segment_examples.append(SegmentationExample(example, segments))
        return segment_examples

    def read_all(self):
        examples = self.docbank.read_all()
        segment_examples = []
        for example in tqdm(examples, desc='Converting:'):
            segments = Segmentation.from_example(example)
            segment_examples.append(SegmentationExample(example, segments))
        return segment_examples

    def get_by_filename(self, filename):
        example = self.docbank.get_by_filename(filename)
        segments = Segmentation.from_example(example)
        segment_example = SegmentationExample(example, segments)
        return segment_example

if __name__ == '__main__':
    txt_dir = r'E:\users\minghaoli\DocBank\DocBank_500K_txt'
    img_dir = r'E:\users\minghaoli\DocBank\DocBank_500K_ori_img'
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    docbank = DocBankReader(txt_dir=txt_dir, img_dir=img_dir)
    docbank_segmentation = DocBankSegmentationReader(docbank)

    # examples = docbank.get_by_filename('1.tar_1401.0001.gz_infoingames_without_metric_arxiv_0_ori.jpg')
    segments = docbank_segmentation.sample_n(1000)

    # for segment in segments:

    #     # filename = os.path.basename(example.filepath)
    #     im = segment.plot_bbox()        
    #     im.show()
    #     # im.save(os.path.join(output_dir, filename.replace('_ori.jpg', '_seg.jpg')))
    #     # im = example.plot()
    #     # im.save(os.path.join(output_dir, filename.replace('_ori.jpg', '_exp.jpg')))
    #     # shutil.copy(os.path.join(img_dir, filename), os.path.join(output_dir, filename))
