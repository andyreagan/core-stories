#!/usr/bin/python
usage = """Use after make-single-latex-file.py.

Copies foo-combined.tex to epj-package
and replaces figure names by fig1, fig2, etc.

See end of file for localized pieces."""

# from sys import argv
from subprocess import call
import re
from os.path import isfile,join,dirname,realpath
from os import getcwd
import argparse
parser = argparse.ArgumentParser(description=usage)
parser.add_argument('texfile')
parser.add_argument('output_folder')
parser.add_argument('full_figure_path',type=bool,default=False)
parser.add_argument('--fig_prefix',type=str,default='fig')

citation_re = re.compile(r"\\cite{([\w]+)}")

if __name__ == "__main__":
    argv = parser.parse_args()
    # print(vars(argv))
    texfile = argv.texfile
    output_folder = argv.output_folder
    # set to local for the local dir
    figure_full_path = argv.full_figure_path
    fig_prefix = argv.fig_prefix
    if figure_full_path:
        # this gets the location of this file...
        # cwd = dirname(realpath(__file__))
        cwd = getcwd()
        figure_path = join(cwd,output_folder)
    else:
        figure_path = ""

    # clean out the package directory
    # call(r"\rm epj-package/*",shell=True)

    # find figures

    outfile = join(output_folder,texfile)
    f = open(texfile,"r")
    g = open(outfile,"w")

    figcount = 1
    supplementary = False
    caption = False
    for line in f:
        if r"\setcounter{page}{1}" in line:
            supplementary = True
            figcount = 1
            fig_prefix += "S"
        # strip comments
        # cheap way to remove them anywhere in the line
        line = line.split("%%")[0]
        if r"\includegraphics" in line:
            # print(line)
            # extract the figure
            match = re.search(r"([\w\\\.\]\[\=]+){([\w\.\-/]+)}()",line)
            figure_fullpath = match.groups()[1]
            figure_filename = figure_fullpath.split("/")[-1]
            # print(figure_fullpath)
            # print(figure_filename)
            # make the new filename
            if figcount > 0:
                figure_newfilename = fig_prefix+str(figcount)+"_"+figure_filename
            else:
                figure_newfilename = figure_filename
            # print(figure_newfilename)

            print("cp {} {}".format(figure_fullpath,join(output_folder,figure_newfilename)))
            call("cp {} {}".format(figure_fullpath,join(output_folder,figure_newfilename)),shell=True)
            if not supplementary:
                # if ".png" in figure_newfilename:
                #     figure_name = figure_newfilename.replace(".png","")
                #     call("convert folder/{}.{{png,pdf}}".format(figure_name),shell=True)
                #     call("rm folder/{}.png".format(figure_name),shell=True)
                #     figure_newfilename = figure_name+".pdf"
                # if ".pdf" in figure_newfilename:
                #     figure_name = figure_newfilename.replace(".pdf","")
                #     if not isfile("epj-package/"+figure_name+".tiff"):
                #         call("./tiffify.pl epj-package/{}.{{pdf,tiff}}".format(figure_name),shell=True)
                #     # keep the pdf
                #     # call(r"rm folder/prefixfigfile.pdf",shell=True)
                #     figure_newfilename = figure_name+".tiff"
                # if figcount > 0:
                #     line = "%% "+match.groups()[0]+"{"+figure_newfilename+"}\n"
                # else:
                #     # this is PLoS's header figure...
                #     line = line.replace(figure_fullpath,figure_filename)
                # line = "%% "+match.groups()[0]+"{"+figure_newfilename+"}\n"
                # pass
                line = match.groups()[0]+"{"+join(figure_path,figure_newfilename)+"}\n"
            else:
                # don't worry about converting the ones in the supplementary
                line = match.groups()[0]+"{"+join(figure_path,figure_newfilename)+"}\n"

            # print(line)
            figcount += 1



        # if caption:
        #     caption = False
        #     # print("\\textbf{"+line.rstrip()+"}")
        #     line = "\\textbf{"+line.rstrip()+"}\n"
        # if r"\caption" in line:
        #     # figures =~ s/caption\{(.*?\.[\s\}])/caption{\\textbf{\1} /msg
        #     # print("found a caption")
        #     caption = True
        if supplementary:
            # these will run on all lines in the supp
            # remove supplenmentary citations
            # line = citation_re.sub("\cite{}",line)
            pass
        else:
            # these will run on all lines in the main file
            # line = line.replace("tbp!","h!")
            # line = line.replace("figure*","figure")
            # line = line.replace("table*","table")
            # line = line.replace(r"\section{",r"\section*{")
            # line = line.replace(r"\subsection{",r"\subsection*{")
            pass

        # finally, write the line to a file!
        g.write(line)
    f.close()
    g.close()
