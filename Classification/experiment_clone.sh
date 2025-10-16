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

# # plbart 

python -u plbart_versionall/clone_pure_code.py --model plbart
python -u plbart_versionall/clone_concat.py --model plbart
python -u plbart_versionall/clone_max_pool.py --model plbart
python -u plbart_versionall/clone_diff_concat.py --model plbart

# python -u codet5_callgraph/clone_pure_code.py --model plbart 
python -u codet5_callgraph/clone_concat.py --model plbart
python -u codet5_callgraph/clone_max_pool.py --model plbart
python -u codet5_callgraph/clone_diff_concat.py --model plbart

# python -u codet5_versionall_callgraph/clone_pure_code.py --model plbart 
python -u codet5_versionall_callgraph/clone_concat.py --model plbart
python -u codet5_versionall_callgraph/clone_max_pool.py --model plbart
python -u codet5_versionall_callgraph/clone_diff_concat.py --model plbart

# python -u codet5_versionall_numofdays/clone_pure_code.py --model plbart
python -u codet5_versionall_numofdays/clone_concat.py --model plbart
python -u codet5_versionall_numofdays/clone_max_pool.py --model plbart
python -u codet5_versionall_numofdays/clone_diff_concat.py --model plbart

# python -u codet5_versionall_callgraph_numofdays/clone_pure_code.py --model plbart
python -u codet5_versionall_callgraph_numofdays/clone_concat.py --model plbart
python -u codet5_versionall_callgraph_numofdays/clone_max_pool.py --model plbart
python -u codet5_versionall_callgraph_numofdays/clone_diff_concat.py --model plbart

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
