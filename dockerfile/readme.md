## Docker description

Explain how to build and run docker image to generate whole slide image prediction heatmaps.

### Building the docker:

1. Navigate to the directory containing the Dockerfile: 
quip_classification/dockerfile

2. Run the docker build command:
docker build -t til_pipeline .

### Run the docker:
####Required Folders Description:
**svs:** This folder will hold the .svs files to be processed.  
**patches:** This tiled whole slide image patches will be placed in this folder. After the patch extraction step it will contain subfolders with the filenames of the svs files extracted. The patches will be re-used when run on the same svs file multiple times. Otherwise can be deleted to save space.  
**heatmap_txt:** This will hold the text formatted output predictions.  
**heatmap_jsons:** This will hold the json formatted output predictions.  
**heatmap_txt_binary:** This will hold the text formatted binary (thresholded) predictions.  
**heatmap_json_binary:** This will hold the json formatted binary (thresholded) predictions.  
**log:** This folder will hold the output log files.  

####Required Environment Variable Settings:
**MODEL_CONFIG_FILENAME:** The name of the model configuration file. There are 2 main configurations available:  
*config_incep-mix_test_ext.ini:* The inception-v4 model trained on mix of manual and semi-autoamted labels each from a different set of cancer types.  
*config_vgg-mix_test_ext.ini:* The vgg-16 model trained on mix of manual and semi-autoamted labels each from a different set of cancer types.  

**CUDA_VISIBLE_DEVICES:** The gpu device to use. Default is '0'.  

**HEATMAP_VERSION_NAME:** The version name given to the set of predictions.  

**BINARY_HEATMAP_VERSION_NAME:** The version name given to the set of binary (thresholded) predictions.  


####Command to Execute:
There are several scripts that are useful for generating predictions heatmaps from whole slide images. Replace the placeholder {Command} with any of the script filenames :

1. cleanup_heatmap.sh
Predictions are re-used if available. You can skip this step if you are only continuing the prediction without changing the model configuration. 
To create new predictions on the same svs files (probably using a different model) it is important to clean the previously generated predictions before performing running again using another model. This script also cleans the log folder.

2. svs_2_heatmap.sh:
Runs the heatmap generation *including* patch extraction (tiling)

3. patch_2_heatmap.sh:
Runs the heatmap generation *excluding* patch extraction (tiling)

4. threshold_probability_heatmaps.sh:
Creates binary predictions probability maps using the predefined thresholds saved in the model configuration file.


  

####To Run the Docker:
Run the below command replacing the {placeholders} with appropriate settings:
nvidia-docker run --name test_til_pipeline  -it \
-v {svs folder path}::/root/quip_classification/u24_lymphocyte/data/svs  \
-v {patches folder path}:/root/quip_classification/u24_lymphocyte/data/patches   \
-v {heatmap_txt folder path}:/root/quip_classification/u24_lymphocyte/data/heatmap_txt   \
-v {heatmap_jsons folder path}:/root/quip_classification/u24_lymphocyte/data/heatmap_jsons   \
-v {binary_heatmap_txt folder path}:/root/quip_classification/u24_lymphocyte/data/heatmap_txt_binary   \
-v {binary_heatmap_jsons folder path}:/root/quip_classification/u24_lymphocyte/data/heatmap_jsons_binary   \
-v {log folder path}:/root/quip_classification/u24_lymphocyte/data/log
-e MODEL_CONFIG_FILENAME='{model config file name}'  \
-e CUDA_VISIBLE_DEVICES='{GPU ID}'  \
-e HEATMAP_VERSION_NAME='{heatmap version name}'  \
-e BINARY_HEATMAP_VERSION_NAME='{heatmap version name}'  \
-d til_pipeline:latest  {Command}
 

**This is an example command with some settings:**  

nvidia-docker run --name til_pipeline --rm -it \
-v /nfs/bigbrain/shahira/svs:/root/quip_classification/u24_lymphocyte/data/svs  \
-v  /nfs/bigbrain/shahira/patches:/root/quip_classification/u24_lymphocyte/data/patches   \
-v  /nfs/bigbrain/shahira/til_output/heatmap_txt:/root/quip_classification/u24_lymphocyte/data/heatmap_txt   \
-v  /nfs/bigbrain/shahira/til_output/heatmap_json:/root/quip_classification/u24_lymphocyte/data/heatmap_jsons   \
-v  /nfs/bigbrain/shahira/til_output/heatmap_txt_binary:/root/quip_classification/u24_lymphocyte/data/heatmap_txt_binary   \
-v  /nfs/bigbrain/shahira/til_output/heatmap_json_binary:/root/quip_classification/u24_lymphocyte/data/heatmap_jsons_binary   \
-v  /nfs/bigbrain/shahira/til_output/log:/root/quip_classification/u24_lymphocyte/data/log
-e MODEL_CONFIG_FILENAME='config_incep-mix_test_ext.ini'  \
-e CUDA_VISIBLE_DEVICES='0'  \
-e HEATMAP_VERSION_NAME='lym_vgg-mix_probability'  \
-e BINARY_HEATMAP_VERSION_NAME='lym_vgg-mix_binary'  \
-d til_pipeline:latest \
svs_2_heatmap.sh

