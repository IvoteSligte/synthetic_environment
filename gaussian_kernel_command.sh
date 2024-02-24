cat gaussian_kernel.txt | awk 'BEGIN {x = -2; y = -2} { print "(IVec2::new("x", "y"), "$1 / 273");"; x += 1; if (x > 2) { y += 1; x = -2; } }'
