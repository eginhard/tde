from itertools import chain
import os
import os.path as path
import glob
from collections import namedtuple, Counter, defaultdict
import random

import tde
from tde.util.reader import tokenlists_to_corpus
from tde.util.functions import grouper
from tde.data.interval import Interval
from tde.data.fragment import FragmentToken
from tde.goldset import extract_gold_fragments

FileSet = namedtuple('FileSet', ['phn', 'wrd'])

def split_by(iterable, cond):
    r = []
    for e in iterable:
        if cond(e):
            if r:
                yield r
                r = []
        else:
            r.append(e)
    if r:
        yield r

def write_counts(phn_fragments, wrd_fragments, outdir):
    words = Counter(f.mark for f in chain.from_iterable(wrd_fragments))
    phones = Counter(f.mark for f in chain.from_iterable(phn_fragments))

    with open(path.join(outdir, 'wrd.stats'), 'w') as fid:
        fid.write('\n'.join('{0} {1}'.format(k, v)
                            for k, v in words.iteritems()))
    with open(path.join(outdir, 'phn.stats'), 'w') as fid:
        fid.write('\n'.join('{0} {1}'.format(k, v)
                            for k, v in phones.iteritems()))

def load_alignment(fname, strip_tags=True):
    """Loads a .ctm alignment file into FragmentTokens."""
    fragment_lists = []
    fragments = []
    previous_name = ""
    
    for line in open(fname):
        name, _, start, duration, mark = line.strip().split(' ')
        if name != previous_name:
            if fragments != []:
                fragment_lists.append(fragments)
            fragments = []
        previous_name = name
        start = round(float(start), 2)
        stop = start + round(float(duration), 2)
        interval = Interval(start, stop)

        if "phone" in fname and strip_tags:
            mark = mark.split('_')[0]
        
        fragment = FragmentToken(name, interval, mark)
        fragments.append(fragment)
    if fragments != []:
        fragment_lists.append(fragments)
        
    # Phone and word alignments aren't necessarily in the same order, so sort.
    fragment_lists.sort()
    return fragment_lists

def is_contiguous(tokens):
    """Checks whether all intervals in the iterable are contiguous."""
    return all(f1.interval.end == f2.interval.start
               for f1, f2 in zip(tokens, tokens[1:]))

def load_filesets(phnfile, wrdfile):
    """Loads the phone and word alignment files."""
    fragments = []
    phn_fragment_lists = load_alignment(phnfile)
    wrd_fragment_lists = load_alignment(wrdfile)

    # Check that the number of phone and word alignments match.
    assert (len(phn_fragment_lists) == len(wrd_fragment_lists))

    for phn_fragments, wrd_fragments in zip(phn_fragment_lists,
                                            wrd_fragment_lists):
        # if is_contiguous(phn_fragments) and is_contiguous(wrd_fragments):
        fragments.append(FileSet(phn_fragments, wrd_fragments))
    return fragments

def load(phnfile, wrdfile, outdir):
    fragments = load_filesets(phnfile, wrdfile)
    phn_fragments, wrd_fragments = zip(*fragments)

    # remove "sil", "spn_S", "<unk>"
    phn_fragments = [[f for f in fl if not f.mark in ['sil', 'spn_S', 'spn']]
                     for fl in phn_fragments]
    wrd_fragments = [[f for f in fl if not f.mark in ['<unk>']]
                     for fl in wrd_fragments]

    intervals_from_phn = {fl[0].name: Interval(fl[0].interval.start,
                                               fl[-1].interval.end)
                          for fl in phn_fragments}
    intervals_from_wrd = {fl[0].name: Interval(fl[0].interval.start,
                                               fl[-1].interval.end)
                          for fl in wrd_fragments}
        
    # check that the total file intervals match up
    assert (intervals_from_phn == intervals_from_wrd)
    # check that each word corresponds to a sequence of phones exactly
    wrd_corpus = tokenlists_to_corpus(wrd_fragments)
    phn_corpus = tokenlists_to_corpus(phn_fragments)
    # (will raise exception if exact match is not found)
    (phn_corpus.tokens_exact(name, interval)
     for name, interval, mark in wrd_corpus.iter_fragments())

    # write concatenated phn, wrd files
    with open(path.join(outdir, 'globalphone.phn'), 'w') as fp:
        for fragment in sorted(chain.from_iterable(phn_fragments),
                               key=lambda x: (x.name, x.interval.start)):
            fp.write('{0} {1:.2f} {2:.2f} {3}\n'.format(
                fragment.name, fragment.interval.start, fragment.interval.end,
                fragment.mark))
    with open(path.join(outdir, 'globalphone.wrd'), 'w') as fp:
        for fragment in sorted(chain.from_iterable(wrd_fragments),
                               key=lambda x: (x.name, x.interval.start)):
            fp.write('{0} {1:.2f} {2:.2f} {3}\n'.format(
                fragment.name, fragment.interval.start, fragment.interval.end,
                fragment.mark))
    with open(path.join(outdir, 'globalphone.split'), 'w') as fp:
        for name, interval in sorted(intervals_from_phn.iteritems()):
            fp.write('{0} {1:.2f} {2:.2f}\n'.format(name,
                                                    interval.start,
                                                    interval.end))

    return phn_fragments, wrd_fragments

def make_gold(phn_fragments, outdir, n_jobs, verbose):
    pairs = extract_gold_fragments(phn_fragments, verbose=verbose, n_jobs=n_jobs)
    classes = defaultdict(set)
    for fragment in chain.from_iterable(pairs):
        classes[fragment.mark].add(fragment)
    with open(path.join(outdir, 'globalphone.classes'), 'w') as fp:
        for ix, mark in enumerate(sorted(classes.keys())):
            fp.write('Class {0} [{1}]\n'.format(ix, ','.join(mark)))
            for fragment in sorted(classes[mark],
                                   key=lambda x: (x.name,
                                                  x.interval.start)):
                fp.write('{0} {1:.2f} {2:.3f}\n'.format(
                    fragment.name, fragment.interval.start, fragment.interval.end))
    return classes

def split_em(phn_fragments, outdir):
    intervals = {f[0].name: Interval(f[0].interval.start, f[-1].interval.end)
                 for f in phn_fragments}

    names_cross = list(grouper(1000, random.sample(intervals.items(), 4000)))
    intervals_per_speaker = defaultdict(set)
    for fname, interval in intervals.iteritems():
        intervals_per_speaker[fname.split('_')[0]].add((fname, interval))
    names_within = [list(v)
                    for v in intervals_per_speaker.values()
                    if len(v) > 40]

    with open(path.join(outdir, 'globalphone.intervals.cross'), 'w') as fp:
        fp.write('\n\n'.join('\n'.join('{0} {1:.2f} {2:.2f}'.format(
            name, interval.start, interval.end)
                                       for name, interval in sorted(ns))
                             for ns in names_cross))

    with open(path.join(outdir, 'globalphone.intervals.within'), 'w') as fp:
        fp.write('\n\n'.join('\n'.join('{0} {1:.2f} {2:.2f}'.format(
            name, interval.start, interval.end)
                                       for name, interval in sorted(ns))
                             for ns in names_within))
        # fp.write('\n\n'.join('\n'.join(sorted(ns)) for ns in names_within))

    fnames = list(set(f[0].name for f in phn_fragments))
    with open(path.join(outdir, 'globalphone.files'), 'w') as fp:
        fp.write('\n'.join(sorted(fnames)))


if __name__ == '__main__':
    import argparse
    def parse_args():
        parser = argparse.ArgumentParser(
            prog='prep_globalphone.py',
            formatter_class=argparse.RawTextHelpFormatter,
            description='Prep the GlobalPhone corpus for track 2')
        # parser.add_argument('lang', metavar='LANG',
        #                     nargs=1,
        #                     help='2-letter code of language to process')
        parser.add_argument('phnfile', metavar='PHNFILE',
                            nargs=1,
                            help='phone alignment file')
        parser.add_argument('wrdfile', metavar='WRDFILE',
                            nargs=1,
                            help='word alignment file')
        parser.add_argument('outdir', metavar='OUTDIR',
                            nargs=1,
                            help='output directory')
        parser.add_argument('-v', '--verbose',
                            action='store_true',
                            dest='verbose',
                            default=False,
                            help='talk more')
        parser.add_argument('-n', '--n_jobs',
                            action='store',
                            dest='n_jobs',
                            default=1,
                            help='number of parallel jobs')
        return vars(parser.parse_args())
    args = parse_args()
    # lang = args['lang'][0]
    phnfile = args['phnfile'][0]
    wrdfile = args['wrdfile'][0]
    outdir = args['outdir'][0]
    verbose = args['verbose']
    n_jobs = int(args['n_jobs'])

    if verbose:
        print 'loading files'
    phn_fragments, wrd_fragments = load(phnfile, wrdfile, outdir)

    if verbose:
        print 'extracting gold'
    clsdict = make_gold(phn_fragments, outdir, n_jobs, verbose)

    if verbose:
        print 'splitting folds'
    split_em(phn_fragments, outdir)

    if verbose:
        print 'all done.'
