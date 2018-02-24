using GraphLayout

# loc_x, loc_y = layout_spring_adj(small)

# draw_layout_adj(small, loc_x, loc_y, filename="test1.svg")

println("connect only to closest neighbor")

closest = zeros(Int64,size(small))

# small += 10000*eye(length(small[:,1]))
for i in 1:length(small[:,1])
    small[i,i] = 10000
end
for i in 1:length(closest[1,:])
    ind = indmin(small[:,i])
    closest[ind,i] = 1
    # closest[i,ind] = 1
end

loc_x, loc_y = layout_spring_adj(closest)

draw_layout_adj(closest, loc_x, loc_y, filename="test2.svg")
