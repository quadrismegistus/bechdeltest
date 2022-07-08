# Constants
import os,warnings,sys
warnings.filterwarnings('ignore')
PATH_CODE = os.path.dirname(os.path.realpath(__file__))
PATH_HOME = HOME = os.path.expanduser('~')
PATH_ROOT = os.path.join(PATH_HOME,'.bechdel')
PATH_DATA = os.path.join(PATH_ROOT,'data')
PATH_CORPUS = os.path.join(PATH_DATA,'corpus')
PATH_DATA_SCRIPTS_TXT = os.path.join(PATH_DATA, 'scripts_txt')
PATH_DATA_SCRIPTS_PARSED = os.path.join(PATH_DATA, 'scripts_parsed')
PATH_CORPUS_METADATA = os.path.join(PATH_CORPUS, 'metadata.csv')
PATH_CORPUS_TEXTS = os.path.join(PATH_CORPUS, 'texts')

# native imports
import os,sys,random,json,pickle,shutil
from collections import defaultdict,Counter
from pprint import pprint
from tqdm import tqdm

# external imports
import pandas as pd
from yapmap import pmap,pmap_iter


# local
from .tools import *
from .parsers.names import *
from .parsers.texts import *
from .parsers.scripts import *
from .parsers.convos import *
from .parsers.casts import *