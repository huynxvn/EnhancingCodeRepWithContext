# # #!/bin/bash

# # codebert 
python -u transformer_versionall/clone_pure_code.py --model codebert
python -u transformer_versionall/clone_concat.py --model codebert
python -u transformer_versionall/clone_max_pool.py --model codebert
python -u transformer_versionall/clone_diff_concat.py --model codebert

# python -u transformer_callgraph/clone_pure_code.py --model codebert 
python -u transformer_callgraph/clone_concat.py --model codebert 
python -u transformer_callgraph/clone_max_pool.py --model codebert 
python -u transformer_callgraph/clone_diff_concat.py --model codebert

# python -u transformer_versionall_callgraph/clone_pure_code.py --model codebert 
python -u transformer_versionall_callgraph/clone_concat.py --model codebert 
python -u transformer_versionall_callgraph/clone_max_pool.py --model codebert 
python -u transformer_versionall_callgraph/clone_diff_concat.py --model codebert

# python -u transformer_versionall_numofdays/clone_pure_code.py --model codebert 
python -u transformer_versionall_numofdays/clone_concat.py --model codebert 
python -u transformer_versionall_numofdays/clone_max_pool.py --model codebert 
python -u transformer_versionall_numofdays/clone_diff_concat.py --model codebert

# python -u transformer_versionall_callgraph_numofdays/clone_pure_code.py --model codebert 
python -u transformer_versionall_callgraph_numofdays/clone_concat.py --model codebert 
python -u transformer_versionall_callgraph_numofdays/clone_max_pool.py --model codebert 
python -u transformer_versionall_callgraph_numofdays/clone_diff_concat.py --model codebert


# graphcodebert 
python -u transformer_versionall/clone_pure_code.py --model graphcodebert
python -u transformer_versionall/clone_concat.py --model graphcodebert
python -u transformer_versionall/clone_max_pool.py --model graphcodebert
python -u transformer_versionall/clone_diff_concat.py --model graphcodebert

# python -u transformer_callgraph/clone_pure_code.py --model graphcodebert 
python -u transformer_callgraph/clone_concat.py --model graphcodebert 
python -u transformer_callgraph/clone_max_pool.py --model graphcodebert 
python -u transformer_callgraph/clone_diff_concat.py --model graphcodebert

# python -u transformer_versionall_callgraph/clone_pure_code.py --model graphcodebert 
python -u transformer_versionall_callgraph/clone_concat.py --model graphcodebert 
python -u transformer_versionall_callgraph/clone_max_pool.py --model graphcodebert 
python -u transformer_versionall_callgraph/clone_diff_concat.py --model graphcodebert

# python -u transformer_versionall_numofdays/clone_pure_code.py --model graphcodebert 
python -u transformer_versionall_numofdays/clone_concat.py --model graphcodebert 
python -u transformer_versionall_numofdays/clone_max_pool.py --model graphcodebert 
python -u transformer_versionall_numofdays/clone_diff_concat.py --model graphcodebert

# python -u transformer_versionall_callgraph_numofdays/clone_pure_code.py --model graphcodebert 
python -u transformer_versionall_callgraph_numofdays/clone_concat.py --model graphcodebert 
python -u transformer_versionall_callgraph_numofdays/clone_max_pool.py --model graphcodebert 
python -u transformer_versionall_callgraph_numofdays/clone_diff_concat.py --model graphcodebert

# # codet5base 

python -u codet5_versionall/clone_pure_code.py --model codet5base
python -u codet5_versionall/clone_concat.py --model codet5base
python -u codet5_versionall/clone_max_pool.py --model codet5base
python -u codet5_versionall/clone_diff_concat.py --model codet5base

# python -u codet5_callgraph/clone_pure_code.py --model codet5base 
python -u codet5_callgraph/clone_concat.py --model codet5base 
python -u codet5_callgraph/clone_max_pool.py --model codet5base 
python -u codet5_callgraph/clone_diff_concat.py --model codet5base

# python -u codet5_versionall_callgraph/clone_pure_code.py --model codet5base 
python -u codet5_versionall_callgraph/clone_concat.py --model codet5base 
python -u codet5_versionall_callgraph/clone_max_pool.py --model codet5base 
python -u codet5_versionall_callgraph/clone_diff_concat.py --model codet5base

# python -u codet5_versionall_numofdays/clone_pure_code.py --model codet5base 
python -u codet5_versionall_numofdays/clone_concat.py --model codet5base 
python -u codet5_versionall_numofdays/clone_max_pool.py --model codet5base 
python -u codet5_versionall_numofdays/clone_diff_concat.py --model codet5base

# python -u codet5_versionall_callgraph_numofdays/clone_pure_code.py --model codet5base 
python -u codet5_versionall_callgraph_numofdays/clone_concat.py --model codet5base 
python -u codet5_versionall_callgraph_numofdays/clone_max_pool.py --model codet5base 
python -u codet5_versionall_callgraph_numofdays/clone_diff_concat.py --model codet5base

# astnn
python -u astnn_versionall/clone_pure_code.py 
python -u astnn_versionall/clone_concat.py 
python -u astnn_versionall/clone_max_pool.py 
python -u astnn_versionall/clone_diff_concat.py 

# python -u astnn_callgraph/clone_pure_code.py 
python -u astnn_callgraph/clone_concat.py 
python -u astnn_callgraph/clone_max_pool.py 
python -u astnn_callgraph/clone_diff_concat.py 

# python -u astnn_versionall_callgraph/clone_pure_code.py 
python -u astnn_versionall_callgraph/clone_concat.py 
python -u astnn_versionall_callgraph/clone_max_pool.py 
python -u astnn_versionall_callgraph/clone_diff_concat.py

# python -u astnn_versionall_numofdays/clone_pure_code.py 
python -u astnn_versionall_numofdays/clone_concat.py 
python -u astnn_versionall_numofdays/clone_max_pool.py 
python -u astnn_versionall_numofdays/clone_diff_concat.py 

# python -u astnn_versionall_callgraph_numofdays/clone_pure_code.py 
python -u astnn_versionall_callgraph_numofdays/clone_concat.py 
python -u astnn_versionall_callgraph_numofdays/clone_max_pool.py 
python -u astnn_versionall_callgraph_numofdays/clone_diff_concat.py

# # classification

# codebert
python -u transformer_versionall/class_pure_code.py --model codebert
python -u transformer_versionall/class_concat.py --model codebert 
python -u transformer_versionall/class_max_pool.py --model codebert

# python -u transformer_callgraph/class_pure_code.py --model codebert 
python -u transformer_callgraph/class_concat.py --model codebert
python -u transformer_callgraph/class_max_pool.py --model codebert 

# python -u transformer_versionall_callgraph/class_pure_code.py --model codebert 
python -u transformer_versionall_callgraph/class_concat.py --model codebert 
python -u transformer_versionall_callgraph/class_max_pool.py --model codebert 

# python -u transformer_versionall_numofdays/class_pure_code.py --model codebert 
python -u transformer_versionall_numofdays/class_concat.py --model codebert 
python -u transformer_versionall_numofdays/class_max_pool.py --model codebert 

# python -u transformer_versionall_callgraph_numofdays/class_pure_code.py --model codebert 
python -u transformer_versionall_callgraph_numofdays/class_concat.py --model codebert 
python -u transformer_versionall_callgraph_numofdays/class_max_pool.py --model codebert

# graphcodebert
python -u transformer_versionall/class_pure_code.py --model graphcodebert
python -u transformer_versionall/class_concat.py --model graphcodebert 
python -u transformer_versionall/class_max_pool.py --model graphcodebert

# python -u transformer_callgraph/class_pure_code.py --model graphcodebert 
python -u transformer_callgraph/class_concat.py --model graphcodebert
python -u transformer_callgraph/class_max_pool.py --model graphcodebert 

# python -u transformer_versionall_callgraph/class_pure_code.py --model graphcodebert 
python -u transformer_versionall_callgraph/class_concat.py --model graphcodebert 
python -u transformer_versionall_callgraph/class_max_pool.py --model graphcodebert 

# python -u transformer_versionall_numofdays/class_pure_code.py --model graphcodebert 
python -u transformer_versionall_numofdays/class_concat.py --model graphcodebert 
python -u transformer_versionall_numofdays/class_max_pool.py --model graphcodebert 

# python -u transformer_versionall_callgraph_numofdays/class_pure_code.py --model graphcodebert 
python -u transformer_versionall_callgraph_numofdays/class_concat.py --model graphcodebert 
python -u transformer_versionall_callgraph_numofdays/class_max_pool.py --model graphcodebert

# # codet5base
python -u codet5_versionall/class_pure_code.py --model codet5base
python -u codet5_versionall/class_concat.py --model codet5base 
python -u codet5_versionall/class_max_pool.py --model codet5base

# python -u codet5_callgraph/class_pure_code.py --model codet5base 
python -u codet5_callgraph/class_concat.py --model codet5base
python -u codet5_callgraph/class_max_pool.py --model codet5base 

# python -u codet5_versionall_callgraph/class_pure_code.py --model codet5base 
python -u codet5_versionall_callgraph/class_concat.py --model codet5base 
python -u codet5_versionall_callgraph/class_max_pool.py --model codet5base 

# python -u codet5_versionall_numofdays/class_pure_code.py --model codet5base 
python -u codet5_versionall_numofdays/class_concat.py --model codet5base 
python -u codet5_versionall_numofdays/class_max_pool.py --model codet5base 

# python -u codet5_versionall_callgraph_numofdays/class_pure_code.py --model codet5base 
python -u codet5_versionall_callgraph_numofdays/class_concat.py --model codet5base 
python -u codet5_versionall_callgraph_numofdays/class_max_pool.py --model codet5base

# # astnn
python -u astnn_versionall/class_pure_code.py 
python -u astnn_versionall/class_concat.py 
python -u astnn_versionall/class_max_pool.py 

# python -u astnn_callgraph/class_pure_code.py 
python -u astnn_callgraph/class_concat.py 
python -u astnn_callgraph/class_max_pool.py 

# python -u astnn_versionall_callgraph/clone_pure_code.py 
python -u astnn_versionall_callgraph/class_concat.py 
python -u astnn_versionall_callgraph/class_max_pool.py 

# python -u astnn_versionall_numofdays/clone_pure_code.py 
python -u astnn_versionall_numofdays/class_concat.py 
python -u astnn_versionall_numofdays/class_max_pool.py 

# python -u astnn_versionall_callgraph_numofdays/class_pure_code.py 
python -u astnn_versionall_callgraph_numofdays/class_concat.py 
python -u astnn_versionall_callgraph_numofdays/class_max_pool.py 
