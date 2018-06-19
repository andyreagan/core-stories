#!/usr/bin/python
usage = """input-bbl.py input.tex output.tex

puts input.bbl inside of input.tex, writing to output.tex"""

import sys
import re

if __name__ == "__main__":

    infile = sys.argv[1]
    bblfile = infile.replace(".tex","")+".bbl"
    outfile = sys.argv[2]
    print("infile={}".format(infile))
    print("bblfile={}".format(bblfile))
    print("outfile={}".format(outfile))

    full_file = open(infile,"r").read()
    bib = open(bblfile,"r").read()
    
    # full_file = re.sub("\\\\bibliography{[\\w]+}",re.escape(bib),full_file)
    match = re.findall("\\\\bibliography{[\\w]+}",full_file)[0]

    open(outfile,"w").write(full_file.replace(match,bib))


