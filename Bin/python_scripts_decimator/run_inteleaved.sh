export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

python interleave_decimate_optimize.py \
  --source_mesh data/nerf_lego_dense_mesh.ply \
  --optimize_source_mesh data/nerf_lego_dense_mesh.ply \
  --output_folder output-final \
  --decimate_steps 10 \
  --optimize_steps 10 \
  --target_error 5.0 \
  --target_num_faces 20 \
  --target_num_vertices 150 \
  --max_edge_length_alpha 2.5 \
  --global_scale 0.0629838 \
  --lr 0.005 \
  --mvc_weight 100 \
  --l2_weight 1e-4

python interleave_decimate_optimize.py \
  --source_mesh data/nerf_lego_dense_mesh.ply \
  --optimize_source_mesh data/nerf_lego_dense_mesh.ply \
  --output_folder output-final-decimate-only \
  --decimate_steps 10 \
  --optimize_steps 0 \
  --target_error 5.0 \
  --target_num_faces 20 \
  --target_num_vertices 150 \
  --max_edge_length_alpha 2.5 \
  --global_scale 0.0629838 \
  --lr 0.005 \
  --mvc_weight 100 \
  --l2_weight 1e-4