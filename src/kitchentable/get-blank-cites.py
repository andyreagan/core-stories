#!/usr/bin/python
usage = """get-blank-cites.py input.tex output.tex"""

import sys
import re
citation_re = re.compile(r"\\cite{([\w]+)")

if __name__ == "__main__":
    inputfile = sys.argv[1]
    outputfile = sys.argv[2]
    fulltext = open(inputfile,"r").read()
    all_matches = citation_re.findall(fulltext)
    print(all_matches)
    print(len(all_matches))
    uniq_matches = set(all_matches)
    print(uniq_matches)
    print(len(uniq_matches))
    nocite = "\\nocite{{{}}}\n".format(",".join(uniq_matches))
    open(outputfile,"w").write(nocite)


