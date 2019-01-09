# Patch extraction

Assumes you want to run this software on bridges AI, instructions are as follows:  

First, build a singularity image with openslide according to build_singularity.sh. You have to do this on a machine with sudo privilege.  
Then download WSIs using list_wsi_on_gcloud_download.sh.  
Finally run patch extraction with:  
sbatch extract_process_submit_jobs.sh
