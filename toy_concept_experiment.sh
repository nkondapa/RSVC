
python extract_model_activations.py --model "nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586" --model_ckpt "./checkpoints/nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --feature_layer_version v1 --output_root ./ --dataset nabirds_modified --end_class_idx 1

python extract_model_activations.py --model "nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586" --model_ckpt "./checkpoints/nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --feature_layer_version v1 --output_root ./ --dataset nabirds_modified --end_class_idx 1

python extract_concepts.py --model "nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586" --model_ckpt "./checkpoints/nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --feature_layer_version v1 --output_root ./ --dataset nabirds_modified --start_class_idx 0 --end_class_idx 1

python extract_concepts.py --model "nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586" --model_ckpt "./checkpoints/nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --feature_layer_version v1 --output_root ./ --dataset nabirds_modified --start_class_idx 0 --end_class_idx 1

python compare_models.py --comparison_config ./comparison_configs/r18_mod0=0.7_test0_vs_r18_moda=0.5_test0-nmf=10_seed=0.json \
 --model_0 "nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586" --model_ckpt_0 "./checkpoints/nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --model_1 "nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586" --model_ckpt_1 "./checkpoints/nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --decomp_method_0 nmf --decomp_method_1 nmf \
 --dataset_0 nabirds_modified --dataset_split_0 "train" --dataset_1 nabirds_modified --dataset_split_1 "train"\
 --feature_layer_version_0 v1 --feature_layer_version_1 v1 \
 --concept_root_folder_0 ./ --concept_root_folder_1 ./ \
 --start_class_idx 0 --end_class_idx 1 --folder_exists overwrite


python evaluate_regression.py --comparison_config ./comparison_configs/r18_mod0=0.7_test0_vs_r18_moda=0.5_test0-nmf=10_seed=0.json \
 --model_0 "nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586" --model_ckpt_0 "./checkpoints/nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --model_1 "nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586" --model_ckpt_1 "./checkpoints/nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --decomp_method_0 nmf --decomp_method_1 nmf \
 --feature_layer_version_0 v1 --feature_layer_version_1 v1 \
 --dataset_0 nabirds_modified --dataset_split_0 "train" --dataset_1 nabirds_modified --dataset_split_1 "train"\
 --concept_root_folder_0 ./ --concept_root_folder_1 ./ \
  --start_class_idx 0 --end_class_idx 1 \
 --eval_dataset nabirds_modified --patchify --data_split test

python visualize_concept_comparison.py --comparison_config ./comparison_configs/r18_mod0=0.7_test0_vs_r18_moda=0.5_test0-nmf=10_seed=0.json \
 --model_0 "nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586" --model_ckpt_0 "./checkpoints/nabirds_mod0=0.7_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --model_1 "nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586" --model_ckpt_1 "./checkpoints/nabirds_mod_all=0.5_test0_r18_fs_a3_seed=4834586/last.ckpt" \
 --decomp_method_0 nmf --decomp_method_1 nmf \
 --feature_layer_version_0 v1 --feature_layer_version_1 v1 \
 --dataset_0 nabirds_modified --dataset_split_0 "train" --dataset_1 nabirds_modified --dataset_split_1 "train"\
 --concept_root_folder_0 ./ --concept_root_folder_1 ./ \
  --start_class_idx 0 --end_class_idx 1 \
  --selected_samples_json selected_samples/toy_concept.json \
 --eval_dataset nabirds_modified --patchify --visualize_concepts --data_split test --sim_pair_output_folder concept_comparison_visualizations
