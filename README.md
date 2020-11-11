# DocBank Loader

DocBank Loader is a dataset loader for DocBank, and can convert DocBank to the Object Detection models' format.

The DocBank GitHub repositoriy is [https://github.com/doc-analysis/DocBank](https://github.com/doc-analysis/DocBank)

## Usage

### Dependency

```
pip install -U Pillow tqdm
```

### Installation

```
git clone https://github.com/doc-analysis/DocBankLoader.git
cd DocBankLoader
pip install .
```

### DocBankLoader

```
from docbank_loader import DocBankLoader

txt_dir = '/path/to/DocBank_500K_txt'
img_dir = '/path/to/DocBank_500K_ori_img'
loader = DocBankLoader(txt_dir=txt_dir, img_dir=img_dir)

# Load all the examples
examples = loader.read_all() 
# <list of docbank_loader.docbank_loader.Example>

# Sample N examples
examples = loader.sample_n(n=5) 
# <list of docbank_loader.docbank_loader.Example>

# Load example by basename/filename
example = loader.get_by_filename('295.tar_1712.06217.gz_main_6_ori.jpg') 
# <docbank_loader.docbank_loader.Example>

# Load examples by the index file
examples = loader.read_by_index('path/to/index/file')
# <list of docbank_loader.docbank_loader.Example>
```

### DocBankConverter

```
from docbank_loader import DocBankLoader, DocBankConverter

txt_dir = '/path/to/DocBank_500K_txt'
img_dir = '/path/to/DocBank_500K_ori_img'
loader = DocBankLoader(txt_dir=txt_dir, img_dir=img_dir)

converter = DocBankConverter(loader)

# Convert all the examples
examples = converter.read_all() 
# <list of docbank_loader.docbank_converter.CVExample>

# Sample N examples and convert
examples = converter.sample_n(n=5) 
# <list of docbank_loader.docbank_converter.CVExample>

# Convert example by basename/filename
example = converter.get_by_filename('295.tar_1712.06217.gz_main_6_ori.jpg') 
# <docbank_loader.docbank_converter.CVExample>
```

### Example

#### Each document page and the corresponding DocBank annotation or CV annotation form an Example.

#### Example for DocBank

```
example = loader.sample_n(n=1)[0]  # Sample a example
type(example)
# <docbank_loader.docbank_loader.Example>

im = example.plot() # Plot the DocBank annotation and return a PIL.Image.Image

bboxes = example.denormalized_bboxes() # return the denormalized bounding boxes of tokens

example.filepath # The image filepath
example.pagesize # The image size
example.words # The tokens
example.bboxes # The normalized bboxes
example.rgbs # The RGB values
example.fontnames # The fontnames
example.structures # The structure labels
```

#### Example for Object Detection models' format

```
example = converter.sample_n(n=1)[0]  # Sample a example
type(example)
# <docbank_loader.docbank_converter.CVExample>

im = example.plot() # Plot the DocBank annotation and return a PIL.Image.Image
im = example.plot_bbox() # Plot the CV annotation and return a PIL.Image.Image
print(example.print_bbox()) # Print the bounding boxes and labels

example.filepath # The image filepath
example.pagesize # The image size
example.bboxes # The bboxes in normalized coordinates
```

