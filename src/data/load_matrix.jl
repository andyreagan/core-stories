println("hello julia")

f = open("analysislogactually.txt","r")

maxread = 14738 #15000

full = zeros(Int64,14738,14738)

# println(full(1,:))
i = 0
for line in readlines(f)
    i+=1
    if i > maxread
        break
    end
    # println(line)
    info = split(strip(line,'\n'),"\t")
    book = info[1]
    g = open(string("sumdistance/",book,".csv"),"r")
    # println(length(readlines(g)))
    distances = map((x)->int(strip(x,'\n')),readlines(g))
    # distances = readlines(g)
    # println(length(distances))
    # println(distances[1:10])
    # println(typeof(distances[1]))
    full[:,i] = distances
    close(g)
end

# println(i)
# println(full[1:100,1:100])

small = full[1:maxread,1:maxread]

close(f)
