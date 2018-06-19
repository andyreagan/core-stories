#!/usr/bin/python
usage = """input-inputs.py input.tex filenamebase output.tex

DOES
1. replaces \inputs{stuff} with stuff.tex
2. clear all comments

DOES NOT
input the bibliography .bib file

notes:
presumes that comments start with two percentage signs: %%
(which works nicely for tab indenting in emacs so you
should be doing it anyway)"""


import sys
import re
filenamebase_input_re = re.compile(r"\s*\\input{\\filenamebase([\.\w]+)")
free_input_re = re.compile(r"\s*\\input{([\.\w/-]+)")

def readlines(filename,basefile):
    outputstring = ""

    print("reading "+filename)
    f = open(filename,"r")
    for line in f:
        # don't need this package anymore
        if r"currfile" in line:
            line = ""
        # clear comments (even at end of lines....)
        if "%%" in line:
            line = line.split("%%")[0]
        if filenamebase_input_re.match(line) is not None:
            input_file = filenamebase_input_re.findall(line)[0]
            input_file_tex = basefile+input_file+".tex"
            g = open(basefile+".inputs.txt","a")
            g.write(input_file.lstrip(".").rstrip(".")+" ")
            g.close()
            outputstring += readlines(input_file_tex,basefile)
        elif free_input_re.match(line) is not None:
            input_file = free_input_re.findall(line)[0]
            input_file_tex = input_file+".tex"
            g = open(basefile+".inputs.txt","a")
            g.write(input_file.lstrip(".").rstrip(".")+" ")
            g.close()
            outputstring += readlines(input_file_tex,basefile)
        else:
            # outputfobj.write(line)
            outputstring += (line)
    f.close()

    return outputstring

if __name__ == "__main__":

    infile = sys.argv[1]
    filenamebase = sys.argv[2]
    outfile = sys.argv[3]
    print("infile={}".format(infile))
    print("filenamebase={}".format(filenamebase))
    print("outfile={}".format(outfile))

    full_file = readlines(infile,filenamebase)

    # # replace 3 newlines with 2
    # full_file = re.sub("\n\\s*\n\\s*\n","\n\n",full_file)
    open(outfile,"w").write(full_file)
