## Generate Dummy Heatmaps ##

Generate checkboard heatmap files to visualize the patches and facilitate the annotation.

In the file ```call_generate_dummy_heatmaps.py``` modify the following variables:

```svs_dir```: svs files parent directory  
```images_csv_filepath```: The file path of the csv file containing the list of files to process. It has the same format as the sample ./images.csv  
```out_dir```: output path

entries for svs files that are not found are logged in the output file: ```failed_log.txt``` 
 
  