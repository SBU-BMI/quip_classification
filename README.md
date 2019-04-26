# Classification Codes

The folders are organized as follows:

**dockerfile:** contains the dockerfile used to build the docker image for the generation of tumor infliltrating lymphocytes prediction heatmaps of whole slide images.

**NNFramework_TF** and **NNFramework_TF_external_call**: contain the codes for training and patch prediction.

**process_results:** contains the scripts that were used in post processing the predictions to generate the results in the paper.

**training_patch_extraction:** contains the code used to generate training datasets.

**u24_lymphocyte:** is the code for whole slide image (WSI) prediction. It can perform both slide tiling and prediction and is used by the docker. It is self contained and does not require any other codes. 




