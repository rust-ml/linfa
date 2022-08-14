set style increment user
set style line 1 lc rgb 'red'
set style line 2 lc rgb 'blue'
set style line 3 lc rgb 'green'

set style data points
plot 'examples/mnist.dat' using 1:2:3 linecolor variable pt 7 ps 2 t ''
