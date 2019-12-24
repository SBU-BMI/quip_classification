function get_patch_coordinates(annotat_file, svs_name, username, image_path, tot_width, tot_height, mpp)

patch_size = get_patch_size(mpp);

calc_width = int64(floor(tot_width / patch_size) * patch_size);
calc_height = int64(floor(tot_height / patch_size) * patch_size);

pred = zeros(floor(tot_height / patch_size), floor(tot_width / patch_size));

annotat = imread(annotat_file);
truth = (annotat(:, :, 1) >= 250);

neg = (annotat(:, :, 1) <= 10);
pos = (annotat(:, :, 1) >= 250);
allcases = neg + pos;
[xs, ys] = find(allcases > 0.5);

fid = fopen(['./patch_coordinates/', svs_name, '.', username, '.txt'], 'w');
for i = 1:length(xs)
    fprintf(fid, '%s\t%s\t%.8f\t%.8f\t%.3f\t%d\t%d\t%d\t%d\t%d\n', svs_name, username, ...
        (ys(i)-0.5) / size(pred,2), (xs(i)-0.5) / size(pred,1), ...
        pred(xs(i), ys(i)), truth(xs(i), ys(i)), ...
        calc_width, calc_height, tot_width, tot_height);
end
fclose(fid);

end


function patch_size = get_patch_size(mpp)

pw_20X = 100;
mag = 10.0 / mpp;
pw = double(int32(floor(10 * pw_20X * mag / 20))) / 10.0;
patch_size = pw;

end
