from .docbank_reader import DocBankReader
from .docbank_seg_reader import DocBankSegmentationReader
from .reader import Reader

from tqdm import tqdm
import h5py

from multiprocessing import Pool
import os
import numpy as np
import pickle
import logging

logger = logging.getLogger('__name__')


class DocBankCache:
    def __init__(self, reader: Reader, cache_dir):
        self.reader = reader
        self.cache_dir = cache_dir    

    @classmethod
    def cache2example(cls, cache):
            return pickle.loads(cache.tobytes()) 

    def _worker_for_bbox(self, basename):
        example = self.reader.get_by_filename(basename)
        basename = os.path.basename(example.filepath).replace('.txt', '').replace('_ori.jpg', '')
        with open(os.path.join(self.cache_dir, basename + '.txt'), 'w') as fp:
            fp.write(example.print_bbox())
        # im = example.plot_bbox()        
        # im.save(os.path.join(self.cache_dir, basename + '.jpg'))
    
    def dump_to_bbox(self, processes=50, chunksize=100):        
        basename_list =self.reader.basename_list

        if processes > 0:
            logger.info('Start multiprocessing.')
            with Pool(processes=processes) as p, tqdm(total=len(basename_list)) as pbar:
                for example in p.imap_unordered(self._worker_for_bbox, basename_list, chunksize=chunksize):
                    pbar.update()
                    
        else:
            logger.info('Start processing.')
            with tqdm(total=len(basename_list)) as pbar:
                for example in map(self._worker_for_bbox, basename_list):
                    pbar.update()
                    

    def dump(self, filename, processes=50, chunksize=100):
        cache_file = os.path.join(self.cache_dir, filename)
        basename_list =self.reader.basename_list

        if processes > 0:
            logger.info('Start multiprocessing.')
            with Pool(processes=processes) as p, tqdm(total=len(basename_list)) as pbar, h5py.File(cache_file, 'w') as hdf5_f:
                for example in p.imap_unordered(self.reader.get_by_filename, basename_list, chunksize=chunksize):
                    pbar.update()
                    basename = os.path.basename(example.filepath).replace('.txt', '').replace('_ori.jpg', '')
                    data = np.frombuffer(pickle.dumps(example), dtype=np.uint8)
                    hdf5_f.create_dataset(basename, data=data)
        else:
            logger.info('Start processing.')
            with tqdm(total=len(basename_list)) as pbar, h5py.File(cache_file, 'w') as hdf5_f:
                for example in map(self.reader.get_by_filename, basename_list):
                    pbar.update()
                    basename = os.path.basename(example.filepath).replace('.txt', '').replace('_ori.jpg', '')
                    data = np.frombuffer(pickle.dumps(example), dtype=np.uint8)
                    hdf5_f.create_dataset(basename, data=data)

    def load_in_memory(self, filename):
        cache_file = os.path.join(self.cache_dir, filename)
        with h5py.File(cache_file, 'r') as hdf5_f:
            basename_list = sorted(list(hdf5_f.keys()))
            examples = []
            for basename in basename_list:
                data = hdf5_f[basename]
                examples.append(pickle.loads(data[()].tobytes()))

        return examples

    def load(self, filename):
        cache_file = os.path.join(self.cache_dir, filename)
        with h5py.File(cache_file, 'r') as hdf5_f:
            basename_list = sorted(list(hdf5_f.keys()))
            caches = []
            for basename in basename_list:
                cache = hdf5_f[basename][()]
                caches.append(cache)
        return caches

