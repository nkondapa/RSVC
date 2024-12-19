
python extract_model_activations.py --model resnet18.a2_in1k --feature_layer_version v1 --output_root /media/nkondapa/SSD2/RSVC/ --start_class_idx 0 --end_class_idx 5
python extract_model_activations.py --model resnet50.a2_in1k --feature_layer_version v1 --output_root /media/nkondapa/SSD2/RSVC/ --start_class_idx 0 --end_class_idx 5

python extract_concepts.py --model resnet18.a2_in1k --feature_layer_version v1 --output_root /media/nkondapa/SSD2/concept_book/ --start_class_idx 0 --end_class_idx 5
python extract_concepts.py --model resnet50.a2_in1k --feature_layer_version v1 --output_root /media/nkondapa/SSD2/concept_book/ --start_class_idx 0 --end_class_idx 5

python compare_models.py --comparison_config ./comparison_configs/r18_flv=v1_vs_r50_flv=v1-nmf=10_seed=0.json \
--start_class_idx 0 --end_class_idx 5 --folder_exists overwrite \
--concept_root_folder_0 /media/nkondapa/SSD2/concept_book/ --concept_root_folder_1 /media/nkondapa/SSD2/concept_book/