
python extract_model_activations.py --model resnet18.a2_in1k --feature_layer_version v1 --output_root /media/nkondapa/SSD2/RSVC/ --start_class_idx 0 --end_class_idx 5
python extract_model_activations.py --model resnet50.a2_in1k --feature_layer_version v1 --output_root /media/nkondapa/SSD2/RSVC/ --start_class_idx 0 --end_class_idx 5

python extract_concepts.py --model resnet18.a2_in1k --feature_layer_version v1 --output_root /media/nkondapa/SSD2/concept_book/ --start_class_idx 0 --end_class_idx 5
python extract_concepts.py --model resnet50.a2_in1k --feature_layer_version v1 --output_root /media/nkondapa/SSD2/concept_book/ --start_class_idx 0 --end_class_idx 5