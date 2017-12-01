# Reuters-21578 dataset downloader and parser
#
# Author:  Eustache Diemert <eustache@diemert.fr> 
# http://scikit-learn.org/stable/auto_examples/applications/plot_out_of_core_classification.html
#
# Modified by @herrfz, get pandas DataFrame from the orig SGML
# License: BSD 3 clause

from __future__ import print_function

import re
import os.path
import fnmatch
import urllib.request
import tarfile
import itertools
from pandas import DataFrame
import xml.etree.ElementTree

###############################################################################
# Reuters Dataset related routines
###############################################################################


def _not_in_sphinx():
    # Hack to detect whether we are running by the sphinx builder
    return '__file__' in globals()


class ReutersParser(xml.etree.ElementTree.ElementTree):
    def __init__(self, verbose=0):
        xml.etree.ElementTree.ElementTree.__init__(self, verbose)
        self._reset()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass

    def end_reuters(self):
        self.body = re.sub(r'\s+', r' ', self.body)
        self.docs.append({'title': self.title,
                          'body': self.body,
                          'topics': self.topics})
        self._reset()

    def start_title(self, attributes):
        self.in_title = 1

    def end_title(self):
        self.in_title = 0

    def start_body(self, attributes):
        self.in_body = 1

    def end_body(self):
        self.in_body = 0

    def start_topics(self, attributes):
        self.in_topics = 1

    def end_topics(self):
        self.in_topics = 0

    def start_d(self, attributes):
        self.in_topic_d = 1

    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)
        self.topic_d = ""


class ReutersStreamReader():

    """Iterate over documents of the Reuters dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    """

    DOWNLOAD_URL = ('http://archive.ics.uci.edu/ml/machine-learning-databases/'
                    'reuters21578-mld/reuters21578.tar.gz')
    ARCHIVE_FILENAME = 'reuters21578.tar.gz'

    def __init__(self, data_path):
        self.data_path = data_path
        if not os.path.exists(self.data_path):
            return "error"

        
    def iterdocs(self):
        """Iterate doc by doc, yield a dict."""
        for root, _dirnames, filenames in os.walk(self.data_path):
            for filename in fnmatch.filter(filenames, '*.xml'):
                path = os.path.join('/Users/ai/github/7967781-c6218c43516d0a867b81a4c1a8169ae747e5b253/reuters21578/', filename)
                parser = ReutersParser()
                for doc in parser.parse(open(path)):
                    yield doc


def get_minibatch(doc_iter, size):
    """Extract a minibatch of examples, return a tuple X, y.

    Note: size is before excluding invalid docs with no topics assigned.

    """
    data = [('{title}\n\n{body}'.format(**doc), doc['topics'])
            for doc in itertools.islice(doc_iter, size)
            if doc['topics']]
    if not len(data):
        return DataFrame([])
    else:
        return DataFrame(data, columns=['text', 'tags'])