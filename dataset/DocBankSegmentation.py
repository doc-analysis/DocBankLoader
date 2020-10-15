from DocBankReader import DocBankReader, TokenInfo
import numpy as np
import random
from collections import Counter
from PIL import Image


# random.seed(42)
np.random.seed(42)

class Segmentation:
    def __init__(self, infos, structure):
        self.infos = infos
        self.structure = structure

class SegmentationExample:
    def __init__(self, example, segments):
        self.filepath = example.filepath
        self.pagesize = example.pagesize
        self.segments = segments

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

class DocBankSegmentation:
    def __init__(self, docbank):
        self.docbank = docbank

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


    @classmethod
    def get_segmentation(cls, example):
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
                struct_list = [t for t in struct_dict[struct] if flags[t[1]] == 0]

        return segments        

if __name__ == '__main__':
    txt_dir = r'E:\users\minghaoli\DocBank\DocBank_500K_txt'
    img_dir = r'E:\users\minghaoli\DocBank\DocBank_500K_ori_img'
    docbank = DocBankReader(txt_dir=txt_dir, img_dir=img_dir)
    docbank_segmentation = DocBankSegmentation(docbank)

    # examples = docbank.get_by_filename('1.tar_1401.0001.gz_infoingames_without_metric_arxiv_0_ori.jpg')
    examples = docbank.sample_n(1)

    for example in examples:
        segments = docbank_segmentation.get_segmentation(example)       
        seg_example = SegmentationExample(example, segments) 
        im = seg_example.plot()
        # im = example.plot()
        im.show()