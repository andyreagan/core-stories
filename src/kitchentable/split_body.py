# this script takes the body tex as the argument
# and split it into a figures block and non figures block
#
# USAGE
# python split_body.py foo.tex
#
# foo.tex -> foo.nofigures.tex foo.figures.tex foo.tables.tex

from sys import argv

if __name__ == "__main__":
    my_file = argv[1]
    file_split = my_file.split(".")
    figure_file = ".".join(file_split[:-1]+["figures"]+[file_split[-1]])
    caption_file = ".".join(file_split[:-1]+["captions"]+[file_split[-1]])
    table_file = ".".join(file_split[:-1]+["tables"]+[file_split[-1]])
    nofigure_file = ".".join(file_split[:-1]+["nofigures"]+[file_split[-1]])
    nofigure_table_file = ".".join(file_split[:-1]+["nofigurestables"]+[file_split[-1]])

    # array to hold the text from each figure
    figures = []
    tables = []

    f = open(my_file,"r")
    figcount = -1
    figure = False
    tablecount = -1
    table = False
    nofigures = ""
    nofigures_tables = ""
    for line in f:
        # inside of a table or a figure, add to appropriate place
        if table:
            tables[tablecount] += line
            nofigures_tables += line
        if figure:
            figures[figcount] += line
        if r"\begin{figure" in line and not line[0] == "%":
            figcount += 1
            print("found figure {0}".format(figcount))
            figure = True
            figures.append(line)
        elif r"\begin{table" in line and not line[0] == "%":
            tablecount += 1
            print("found table {0}".format(tablecount))
            table = True
            tables.append(line)
            nofigures_tables += line
        elif not figure and not table:
            nofigures += line
            nofigures_tables += line
        if r"\end{figure" in line:
            figure = False
        elif r"\end{table" in line:
            table = False
        
    f.close()

    f = open(figure_file,"w")
    f.write("\n".join([x.replace("tbp!","!htb") for x in figures]))
    f.close()

    f = open(caption_file,"w")
    # sneak it by the comment removal by using a single comment
    f.write("\n".join([x.replace("\\includegraphics","% \\includegraphics").replace("tbp!","!htb") for x in figures]))
    f.close()

    f = open(table_file,"w")
    f.write("\n".join(tables))
    f.close()

    f = open(nofigure_file,"w")
    f.write(nofigures)
    f.close()

    f = open(nofigure_table_file,"w")
    f.write(nofigures_tables)
    f.close()
