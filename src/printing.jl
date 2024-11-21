using Printf

function print_results(method_name::String, x, s, y, iter_count::Int)
    println("\n$method_name, iteration $iter_count")
    @printf("x = [%s]  s = [%s]  y = [%s]\n",
        join(map(xi -> @sprintf("%12.5e", xi), x), ", "),
        join(map(zi -> @sprintf("%12.5e", zi), s), ", "),
        join(map(yi -> @sprintf("%12.5e", yi), y), ", "))
end
