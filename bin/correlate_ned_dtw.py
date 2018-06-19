"""Evaluate Spoken Term Discovery"""

from __future__ import division

import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path as path
import random
import seaborn as sns
import sys

from collections import namedtuple
from itertools import izip
from joblib import Parallel, delayed

from tde.data.fragment import FragmentToken
from tde.data.interval import Interval
from tde.measures.nlp import ned
from tde.util.reader import load_corpus_txt
from tde.util.printing import verb_print, banner

VERSION = "0.2.1"

Match = namedtuple('Match', 'fragment1 fragment2 dtw')

def load_match_file(match_fn, phn_corpus):
    with open(match_fn) as f:
        matches = []
        for line in f:
            # if len(matches) > 5000:
            #     break
            fields = line.strip().split()
            if len(fields) == 2:
                base1, base2 = fields
            elif len(fields) == 6:
                dtw = float(fields[4])
                start1, end1, start2, end2 = map(
                    lambda x: float(x) / 100.0, fields[:4])
                interval1 = Interval(start1, end1)
                interval2 = Interval(start2, end2)
                fragment1 = FragmentToken(
                    base1, interval1, phn_corpus.annotation(base1, interval1))
                fragment2 = FragmentToken(
                    base2, interval2, phn_corpus.annotation(base2, interval2))
                matches.append(Match(fragment1, fragment2, dtw))

        random.shuffle(matches)
        return matches[:100000]

def NED(match):
    mark1, mark2 = match.fragment1.mark, match.fragment2.mark
    if len(mark1) > 0 or len(mark2) > 0:
        return ned(mark1, mark2)
    else:
        return np.nan

def ned_sub(matches, verbose, n_jobs):
    # ned
    with verb_print('  ned: calculating scores', verbose, False, True, False):
        ned_scores = Parallel(n_jobs=n_jobs,
                              verbose=5 if verbose else 0,
                              pre_dispatch='n_jobs')(delayed(NED)
                                                     (match)
                                                     for match in matches)
        dtw_scores = [match.dtw for match in matches]

    scores = zip(ned_scores, dtw_scores)
    print(len(scores))
    scores = filter(lambda x: x[0] != np.nan, scores)
    print(len(scores))
    return zip(*scores)

def parse_args():
    parser = argparse.ArgumentParser(
        prog='correlate_ned_dtw',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Find correlation between NED and DTW scores of UTD matches',
        epilog="""Example usage:

        $ ./correlate_ned_dtw master_match resultsdir/ SP
        """)
    parser.add_argument('match_file', metavar='MATCHFILE', nargs=1,
                        help='UTD master_match file')
    parser.add_argument('outdir', metavar='DESTINATION', nargs=1,
                        help='location for the evaluation results')
    parser.add_argument('lang', metavar='LANG', nargs=1,
                        help='GlobalPhone language to evaluate')
    parser.add_argument('-v', '--verbose',
                        action='store_true',
                        dest='verbose',
                        default=False,
                        help='display progress')
    parser.add_argument('-j', '--n-jobs',
                        action='store',
                        type=int,
                        dest='n_jobs',
                        default=1,
                        help='number of cores to use')
    parser.add_argument('-V', '--version', action='version',
                        version="%(prog)s version {version}".format(version=VERSION))
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()

    verbose = args['verbose']
    n_jobs = args['n_jobs']

    match_fn = args['match_file'][0]
    dest = args['outdir'][0]
    lang = args['lang'][0]

    if getattr(sys, 'frozen', False):
        # frozen
        rdir = path.dirname(sys.executable)
        resource_dir = path.join(rdir, 'resources')
    else:
        # unfrozen
        rdir = path.dirname(path.realpath(__file__))
        resource_dir = path.join(rdir, 'resources')

    prefix = 'globalphone-' + lang
    phn_corpus_file       = path.join(resource_dir, prefix + '.phn')

    if verbose:
        print 'globalphone_eval2 version {0}'.format(VERSION)
        print '----------------------------'
        print 'dataset:     globalphone-' + lang
        print 'inputfile:   {0}'.format(match_fn)
        print 'destination: {0}'.format(dest)
        print

    if verbose:
        print banner('Loading phone corpus.')
    phn_corpus = load_corpus_txt(phn_corpus_file)

    if verbose:
        print banner('Loading matches from master_match.')
    matches = load_match_file(match_fn, phn_corpus)

    ned_scores, dtw_scores = ned_sub(matches, verbose, n_jobs)

    with open(dest, 'w') as f:
        for ned_score, dtw_score in zip(ned_scores, dtw_scores):
            f.write("%.4f %.4f\n" % (ned_score, dtw_score))

    # sns.jointplot(np.array(dtw_scores), np.array(ned_scores), kind='kde')
    # plt.show()
