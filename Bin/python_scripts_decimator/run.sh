export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
python -m optimize_cage_cagenerf \
    --source_cage data/nerf_lego_dense_mesh.ply \
    --template_cage data/nerf_lego_cage.ply \
    --lr 0.005 \
    --nepochs 1000 \
    --mvc_weight 100 \
    --l2_weight 1e-4 \
    --output_folder "output-cbuild2-l2-1e-4"
