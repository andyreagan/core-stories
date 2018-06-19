# make-links-plos.py foo-combined.tex
#
# use after make-single-latex-file.pl
#
# copies foo-combined.tex to epj-package
# and replaces figure names by fig1, fig2, etc.
#
# see end of file for localized pieces

from sys import argv
from subprocess import call
import re
from os.path import isfile
citation_re = re.compile(r"\\cite{([\w]+)}")

if __name__ == "__main__":
    file = argv[1]
   
    # clean out the package directory
    # call(r"\rm epj-package/*",shell=True)

    # find figures

    outfile = "epj-package/"+file
    f = open(file,"r")
    g = open(outfile,"w")

    figcount = 1
    supplementary = False
    caption = False
    for line in f:
        if r"\setcounter{page}{1}" in line:
            supplementary = True
            figcount = 1
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
            if supplementary:
                figure_newfilename = "figS"+str(figcount)+"_"+figure_filename
            elif figcount > 0:
                figure_newfilename = "fig"+str(figcount)+"_"+figure_filename
            else:
                figure_newfilename = figure_filename
            # print(figure_newfilename)
            
            print("cp {} epj-package/{}".format(figure_fullpath,figure_newfilename))
            call("cp {} epj-package/{}".format(figure_fullpath,figure_newfilename),shell=True)
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
                line = "%% "+match.groups()[0]+"{"+figure_newfilename+"}\n"
            else:
                # don't worry about converting the ones in the supplementary
                line = match.groups()[0]+"{"+figure_newfilename+"}\n"
                
            # print(line)
            figcount += 1

        # remove supplementary citations
        # if supplementary:
        #     line = citation_re.sub("\cite{}",line)

        # if caption:
        #     caption = False
        #     # print("\\textbf{"+line.rstrip()+"}")
        #     line = "\\textbf{"+line.rstrip()+"}\n"
        # if r"\caption" in line:
        #     # figures =~ s/caption\{(.*?\.[\s\}])/caption{\\textbf{\1} /msg
        #     # print("found a caption")
        #     caption = True
        # if supplementary:

        if not supplementary:
            line = line.replace("tbp!","h!")
            
        # line = line.replace("figure*","figure")
        # line = line.replace("table*","table")
        if not supplementary:
            line = line.replace(r"\section{",r"\section*{")
            line = line.replace(r"\subsection{",r"\subsection*{")

        g.write(line)
    f.close()
    g.close()
