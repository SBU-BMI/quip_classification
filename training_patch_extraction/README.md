# Patch extraction

This software extracts *semi-automatically* labeled training patches for Tumor Infiltrating Lymphocyte (TIL) classification. Assuming that you want to run this software on bridges AI, instructions are as follows:  

First, build a singularity image with openslide according to build_singularity.sh. You have to do this on a machine with sudo privilege.  
Then download WSIs using list_wsi_on_gcloud_download.sh.  
Finally run patch extraction with:  
sbatch extract_process_submit_jobs.sh

The command above submit a job for running extract_process.py. Please check out the comments in extract_process.py.
