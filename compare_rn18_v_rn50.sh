
python extract_model_activations.py --model resnet18.a2_in1k --feature_layer_version v1 --output_root ./ --start_class_idx 0 --end_class_idx 100
python extract_model_activations.py --model resnet50.a2_in1k --feature_layer_version v1 --output_root ./ --start_class_idx 0 --end_class_idx 100

python extract_concepts.py --model resnet18.a2_in1k --feature_layer_version v1 --output_root ./ --start_class_idx 0 --end_class_idx 100
python extract_concepts.py --model resnet50.a2_in1k --feature_layer_version v1 --output_root ./ --start_class_idx 0 --end_class_idx 100

python compare_models.py --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--start_class_idx 0 --end_class_idx 100 --folder_exists overwrite \
--concept_root_folder_0 ./ --concept_root_folder_1 ./

python evaluate_regression.py --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--concept_root_folder_0 ./ --concept_root_folder_1 ./ \
--start_class_idx 0 --end_class_idx 100 \
--data_split val --patchify

python concept_integrated_gradients.py --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--start_class_idx 0 --end_class_idx 100 \
--model_0 resnet18.a2_in1k --model_1 resnet50.a2_in1k \
--concept_root_folder_0 ./ --concept_root_folder_1 ./

python replacement_test.py --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--start_class_idx 0 --end_class_idx 100 \
--concept_root_folder_0 ./ --concept_root_folder_1 ./

python visualize_similarity_vs_importance.py --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--start_class_idx 0 --end_class_idx 100 \
--concept_root_folder_0 ./ --concept_root_folder_1 ./

## Layerwise comparisons
python compare_models.py --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0_all_layers.json \
--start_class_idx 0 --end_class_idx 100 --folder_exists overwrite \
--concept_root_folder_0 ./ --concept_root_folder_1 ./

# outputs will be in outputs/visualizations/layerwise_concept_comparisons/{comparison_save_name}/{comparison_param_name}/pearson/
# mean_max_similarity_matrix.png is the plot shown in the paper
python visualize_layerwise_comparisons.py --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0_all_layers.json \
--start_class_idx 0 --end_class_idx 100 \
--concept_root_folder_0 ./ --concept_root_folder_1 ./
