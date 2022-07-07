# Constants
import os
PATH_CODE = os.path.dirname(os.path.realpath(__file__))
PATH_HOME = HOME = os.path.expanduser('~')
PATH_ROOT = os.path.join(PATH_HOME,'.bechdel')
PATH_DATA = os.path.join(PATH_ROOT,'data')
PATH_DATA_SCRIPTS_TXT = os.path.join(PATH_DATA, 'scripts_txt')
PATH_DATA_SCRIPTS_PARSED = os.path.join(PATH_DATA, 'scripts_parsed')

# imports
import os,sys,random,json,pickle
import pandas as pd
from pprint import pprint
from collections import defaultdict,Counter


# local
from .parsers.scripts import *
from .parsers.convos import *