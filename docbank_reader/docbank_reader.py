import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

from .reader import Reader
import logging

logger = logging.getLogger('__name__')


class TokenInfo:
    def __init__(self, word, bbox, rgb, fontname, structure):                
        self.word = word
        self.bbox = bbox
        self.rgb = rgb
        self.fontname = fontname
        self.structure = structure
    def __str__(self):
        return '\t'.join([
                          str(self.word),
                          str(self.bbox),
                          str(self.rgb),
                          str(self.fontname),
                          str(self.structure)])

    def __repr__(self):
        return 'TokenInfo({}, {}, {})'.format(self.word, str(self.bbox), self.structure)                          
    @classmethod
    def from_example(cls, example):
        infos = []
        for word, bbox, rgb, fontname, structure in zip(example.words, example.bboxes, example.rgbs, example.fontnames, example.structures):
            infos.append(cls(word, bbox, rgb, fontname, structure))
        return infos

    @classmethod
    def is_neighbor(cls, info0, info1, x_tolerance=15, y_tolerance=16):
        bbox0 = info0.bbox
        bbox1 = info1.bbox

        # y axis
        if bbox1[1] - bbox0[3] > y_tolerance or bbox0[1] - bbox1[3] > y_tolerance:
            return False
        # x axis
        if bbox1[0] - bbox0[2] > x_tolerance or bbox0[0] - bbox1[2] > x_tolerance:
            return False

        return True


class Example:
    def __init__(self, filepath, pagesize, words, bboxes, rgbs, fontnames, structures):
        assert len(words) == len(bboxes)
        assert len(bboxes) == len(rgbs)
        assert len(rgbs) == len(fontnames)
        assert len(fontnames) == len(structures)
        
        self.filepath = filepath
        self.pagesize = pagesize
        self.words = words
        self.bboxes = bboxes
        self.rgbs = rgbs
        self.fontnames = fontnames
        self.structures = structures
        self._infos = None
    def __str__(self):
        return '\n'.join(['Filepath:', self.filepath, 
                          'Pagesize:', str(self.pagesize),
                          'Words:', str(self.words),
                          'Bboxes:', str(self.bboxes),
                          'Rgbs', str(self.rgbs),
                          'Fontnames', str(self.fontnames),
                          'Structures', str(self.structures)])

    @property
    def infos(self):
        if not self._infos:
            self._infos = TokenInfo.from_example(self)
        
        return self._infos

    def plot(self):        
        width, height = self.pagesize
        im = np.zeros(list(self.pagesize) + [3], dtype=np.uint8)
        
        struct_dict = {}
        for info in self.infos:
            struct = info.structure
            if struct in struct_dict:
                struct_dict[struct].append(info)
            else:
                struct_dict[struct] = [info]
                
        for struct in struct_dict.keys():
            color = np.random.randint(256, size=3)
            for info in struct_dict[struct]:                    
                x0, y0, x1, y1 = info.bbox
                x0, y0, x1, y1 = int(x0 * width / 1000), int(y0 * height / 1000), int(x1 * width / 1000), int(
                    y1 * height / 1000)

                for x in range(x0, x1):
                    for y in range(y0, y1):
                        im[x, y] = color

        im = np.swapaxes(im, 0, 1)
        im = Image.fromarray(im, mode='RGB')

        return im

    def denormalized_bboxes(self):
        re = []
        width, height = self.pagesize
        for bbox in self.bboxes:
            deno_bbox = [bbox[0]/1000*width, bbox[1]/1000*height, bbox[2]/1000*width, bbox[3]/1000*height]
            deno_bbox = list(map(int, deno_bbox))
            re.append(deno_bbox)

        return re
        
            

class DocBankReader(Reader):
    def __init__(self, txt_dir, img_dir):
        self.txt_dir = txt_dir
        self.img_dir = img_dir
        
        basename_list = []
        for img_file in tqdm(os.listdir(self.img_dir), desc='Loading file list:'):
            basename = img_file.replace('_ori.jpg', '')
#             txt_file = basename + '.txt'
#             if not os.path.exists(os.path.join(self.txt_dir, txt_file)):
#                 raise NameError('Missing txt file: {}'.format(txt_file))                
            basename_list.append(basename)
        self.basename_list = sorted(basename_list)
                
    def load(self, basename):        
        txt_file = basename + '.txt'
        img_file = basename + '_ori.jpg'
        
        words = []
        bboxes = []
        rgbs = []
        fontnames = []
        structures = []
        
        with open(os.path.join(self.txt_dir, txt_file), 'r', encoding='utf8') as fp:
            for line in fp.readlines():
                tts = line.split()
                if not len(tts) == 10:
                    logger.warning('Incomplete line in file {}'.format(txt_file))
                    continue
                
                word = tts[0]
                bbox = list(map(int, tts[1:5]))
                rgb = list(map(int, tts[5:8]))
                fontname = tts[8]
                structure = tts[9]
                
                words.append(word)
                bboxes.append(bbox)
                rgbs.append(rgb)
                fontnames.append(fontname)
                structures.append(structure)
        
        im = Image.open(os.path.join(self.img_dir, img_file))
        pagesize = im.size
        example = Example(
            filepath = os.path.join(self.img_dir, img_file),
            pagesize = pagesize,
            words = words,
            bboxes = bboxes,
            rgbs = rgbs,
            fontnames = fontnames,
            structures = structures
        )
        return example
    
    def read_all(self):
        examples = []
        for basename in tqdm(self.basename_list, desc='Loading examples:'):
            examples.append(self.load(basename))
        return examples
    
    def sample_n(self, n):
        examples = []
        for basename in tqdm(random.sample(self.basename_list, n), desc='Sampling examples:'):
            examples.append(self.load(basename))            
        return examples

    def get_by_filename(self, filename):
        basename = filename.replace('.txt', '').replace('_ori.jpg', '')
        return self.load(basename)
    
    def read_by_index(self, index_path):
        examples = []
        with open(index_path, 'r') as fp:
            for txt_file in tqdm(fp.readlines(), desc='Loading examples:'):
                txt_file = txt_file.rstrip()
                examples.append(self.get_by_filename(txt_file))
        return examples
    