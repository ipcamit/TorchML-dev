
#!/usr/bin/env bash
set -euo pipefail
echo "Removing torch-scatter-sparse-kim files from ${PREFIX}" 
rm -rf "${PREFIX}/include/torch_sparse" "${PREFIX}/include/torch_scatter"
rm -f  "${PREFIX}/lib/libtorch_sparse"* "${PREFIX}/lib/libtorch_scatter"*
