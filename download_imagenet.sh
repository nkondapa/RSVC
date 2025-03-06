#!/bin/bash
# Download Imagenet
parse_yaml() {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|^\($s\)\($w\)$s:$s\"\(.*\)\"$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p"  $1 |
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]}}
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) {vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, $2, $3);
      }
   }'
}
eval $(parse_yaml paths.yaml "paths_")
mkdir -p "$paths_datasets_imagenet"
cd $paths_datasets_imagenet && wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz
cd $paths_datasets_imagenet && wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar && wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
cd $paths_datasets_imagenet && wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar && mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train && tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar && find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# remove corrupted file (png disguised as jpeg)
rm ${paths_datasets_imagenet}/train/n04266014/n04266014_10835.JPEG


wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz && wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar && wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar