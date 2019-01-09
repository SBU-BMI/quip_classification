# Patch extraction

This software extracts *semi-automatically labeled* training patches for Tumor Infiltrating Lymphocyte (TIL) classification. Manually labeled training patches can be downloaded at:  
http://vision.cs.stonybrook.edu/~lehhou/download/lym_cnn_training_data_formated.zip

Assuming that you want to run this software on bridges AI, instructions are as follows:  

First, build a singularity image with openslide according to build_singularity.sh. You have to do this on a machine with sudo privilege.  
Then, download WSIs using list_wsi_on_gcloud_download.sh. Skip this step if you have WSIs already.  
Then, download the semi-automatically generated Tumor Infiltrating Lymphocyte (TIL) maps in the following link. Skip this step if you have the TIL maps already.  
https://stonybrookmedicine.app.box.com/s/ecr7ba8czvqygw90iym0hwpnprrofoas  

Finally run patch extraction with:  
sbatch extract_process_submit_jobs.sh

The command above submit a job for running extract_process.py. Please check out the comments in extract_process.py.

## Format of training data

For both of the manually labeled patches, and the semi-automatically labeled patches, the label is encoded in the filename of each PNG image file. After splitting the filename (without .png extension), check the last field. 0 means it is not lymphocyte infiltrated. 1 means it is.
