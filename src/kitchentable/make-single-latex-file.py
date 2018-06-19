#!/usr/bin/python
usage = """make-single-latex-file.py filebasename-[revtex4,PLOS,PNAS,main].tex

replaces \inputs{stuff} with stuff.tex,
inputs bbl file, etc.

presumes that comments start with two percentage signs: %%
(which works nicely for tab indenting in emacs so you
should be doing it anyway)

creates foo-combined.tex"""

import sys
import re
input_re = re.compile(r"\s*\\input{([\./]*)(\\filenamebase|)([\./\w-]+)")
include_re = re.compile(r"\s*\\include{([\./]*)(\\filenamebase|)([\./\w-]+)")
from os.path import join

def readlines(filename,basefile):
    outputstring = ""
    # print("reading",filename)
    f = open(filename,"r")
    for line in f:
        # don't need this package
        if r"currfile" in line:
            line = ""
        # clear comments
        elif "%%" in line:
            line = line.split("%%")[0]
        if input_re.match(line) is not None:
            # print(input_re.findall(line))
            this_match = input_re.findall(line)[0]
            input_file_tex = join(this_match[0],this_match[1].replace(r"\filenamebase",basefile)+this_match[2]+".tex")
            # print(input_file_tex)
            # # save this for making a diff file...
            # g = open(basefile+".inputs.txt","a")
            # # g.write(input_file_tex.lstrip(".").rstrip(".")+" ")
            # g.write(input_file_tex.lstrip().rstrip()+" ")
            # g.close()
            outputstring += readlines(input_file_tex,basefile)
        elif include_re.match(line) is not None:
            # print(input_re.findall(line))
            this_match = include_re.findall(line)[0]
            input_file = join(this_match[0],this_match[1].replace(r"\filenamebase",basefile)+this_match[2])
            input_file_tex = input_file+".tex"
            # print(input_file_tex)
            # # save this for making a diff file...
            # g = open(basefile+".inputs.txt","a")
            # # g.write(input_file_tex.lstrip(".").rstrip(".")+" ")
            # g.write(input_file_tex.lstrip().rstrip()+" ")
            # g.close()
            outputstring += r"\include{"+input_file+"-combined}\n"
            # now go get the included file...
            with open(input_file+"-combined.tex","w") as g:
                g.write(readlines(input_file_tex,""))
        else:
            # outputfobj.write(line)
            outputstring += (line)
    f.close()
            
    return outputstring

if __name__ == "__main__":

    infile = sys.argv[1]
    basefile = infile.replace(".tex","")
    bblfile = basefile+".bbl"
    # bblfile = sys.argv[2]
    outfile = basefile+"-combined.tex" 
    print("infile={}".format(infile))
    print("basefile={}".format(basefile))
    print("bblfile={}".format(bblfile))
    print("outfile={}".format(outfile))

    # f = open(outfile,"w")
    # readlines(infile,f)
    for fmt in ["-revtex4","-PLOS","-PNAS","-EPJ","-main"]:
        basefile = basefile.replace(fmt,"")
    full_file = readlines(infile,basefile)

    bib = open(bblfile,"r").read()
    # full_file = re.sub("\\\\bibliography{[\\w]+}",re.escape(bib),full_file)
    match = re.findall("\\\\bibliography{[\\w]+}",full_file)[0]
    print(match)
    full_file = full_file.replace(match,bib)
    # replace 3 newlines with 2
    full_file = re.sub("\n\\s*\n\\s*\n","\n\n",full_file)
    open(outfile,"w").write(full_file)


