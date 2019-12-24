function distribution_8x8_sample()

parent_out_dir = 'super_patch_coords';
src_hard_dir = '/nfs/data02/shared/lehhou/lym_data_for_publication/TIL_maps_after_thres_v1';
src_soft_dir = '/nfs/data02/shared/lehhou/lym_data_for_publication/TIL_maps_before_thres_v1';
group_width = 8;
group_height = 8;

n_picked = 5;       % Number of picked patches for each bin in each cancer type

cancertype_arr = {'blca' 'brca' 'cesc' 'coad' 'luad' 'lusc' 'paad' 'prad' 'read' 'skcm' 'stad' 'ucec' 'uvm'};

for cancer_idx = 1:length(cancertype_arr)

cancer_type = cancertype_arr{cancer_idx};
disp(cancer_type);


out_dir = fullfile(parent_out_dir, cancer_type);
if (exist(out_dir) == 0)
    mkdir(out_dir);
end


%src_img_dir = ['/data08/shared/lehhou/active_learning_osprey/rates-' cancer_type '-all-auto'];
src_img_dir = fullfile(src_hard_dir, cancer_type);
src_soft_img_dir = fullfile(src_soft_dir, cancer_type);
bin_src_imgs = dir(fullfile(src_img_dir, '*.png'));

%parpool(12);

arr_all = [];
coor_ctype = cell(65, 1);


%parfor i_img = 1:length(bin_src_imgs)
for i_img = 1:length(bin_src_imgs)
%for i_img = 1:1
    %disp(i_img);
    slide_name = bin_src_imgs(i_img).name(1:end-4);
    csv_path = fullfile(out_dir, [slide_name '.csv']);
    %fileID = fopen(csv_path, 'w');

    bin_img_name = [bin_src_imgs(i_img).name(1:end-4) '.png'];
    %real_img_name = bin_src_imgs(i_img).name;
    %real_img_path = fullfile(src_img_dir, real_img_name);
    bin_img_path = fullfile(src_img_dir, bin_img_name);
    soft_img_path = fullfile(src_soft_img_dir, bin_img_name);

    %real_img = imread(real_img_path);
    bin_img = imread(bin_img_path);
    soft_img = imread(soft_img_path);

    width = size(bin_img, 2);
    height = size(bin_img, 1);

    arr = [];
    coor_arr = cell(65, 1);
    for iH = 1:group_height:height
        for iW = 1:group_width:width
            boundH = min(iH + group_height - 1, height);
            boundW = min(iW + group_width - 1, width);
            count = 0;
            sum_val = 0;
            sum_soft_val1 = 0;
            sum_soft_val2 = 0;
            for h_idx = iH:boundH
                for w_idx = iW:boundW
                    % real value
                    %real_value = double(real_img(h_idx, w_idx, 1)) / 255.0;

                    % bin value
                    bin_value = 0;
                    % if this is pos tile
                    if (bin_img(h_idx, w_idx, 1) > bin_img(h_idx, w_idx, 3))
                        bin_value = 1;
                        count = count + 1;
                    end

                    soft_value1 = 0;
                    if (soft_img(h_idx, w_idx, 1) > 58)
                        soft_value1 = 1;
                    end
                    soft_value2 = 0;
                    if (soft_img(h_idx, w_idx, 1) > 127)
                        soft_value2 = 1;
                    end

                    sum_val = sum_val + bin_value;
                    sum_soft_val1 = sum_soft_val1 + soft_value1;
                    sum_soft_val2 = sum_soft_val2 + soft_value2;

                    % if this is a tissue tile
                    if (bin_img(h_idx, w_idx, 3) > 128)
                        count = count + 1;
                    end
                    %count = count + 1;
                end
            end

            %ratio = double(sum_val) / double(count);
            if (count ~= 0)
                ratio = double(sum_val) / double(count) * double(group_width) * double(group_height);
                bin_idx = floor(ratio);
                soft_idx1 = floor(double(sum_soft_val1) / double(count) * double(group_width) * double(group_height));
                soft_idx2 = floor(double(sum_soft_val2) / double(count) * double(group_width) * double(group_height));
                coor_arr{bin_idx + 1} = [coor_arr{bin_idx + 1}; [iH, iW, boundH, boundW, soft_idx1+1, soft_idx2+1]];
                %ratio = double(sum_val);
                %disp(ratio);

                %{
                if (ratio > 0)
                    disp('kk');
                    disp(sum_val);
                    disp(count);
                    disp(ratio);
                end
                %}
                %x_center = int64((iW + boundW) / 2);
                %y_center = int64((iH + boundH) / 2);
                %fprintf(fileID, '%d,%d,%.4f\n', x_center, y_center, ratio);

                arr = [arr ratio];
                arr_all = [arr_all ratio];
            end
        end
    end

    %disp(max(arr(:)))
    %[hist_count, hist_value] = hist(arr, [0:64]);
    %csvwrite(csv_path, [hist_value', hist_count'])

    for i = 1:65
        coor = coor_arr{i};

        id_arr_shuf = randperm(size(coor, 1));
        id_picked = id_arr_shuf(1:min(n_picked, length(id_arr_shuf)));
        coor_picked = coor(id_picked, :);

        slide_name_cell = cell(size(coor_picked, 1), 1);
        slide_name_cell(:) = {slide_name};
        written_line = [slide_name_cell, num2cell(coor_picked)];
        %disp(written_line);
        coor_ctype{i} = [coor_ctype{i}; written_line];
    end

    %fclose(fileID);
end

%all_csv_path = fullfile(out_dir, 'all.csv');
%[hist_count_all, hist_value_all] = hist(arr_all, [0:64]);
%csvwrite(all_csv_path, [hist_value_all', hist_count_all'])

for i = 1:65
    %file_bin_name = fullfile(out_dir, [num2str(i-1) '.txt']), coor_ctype{i});
    bin_info = coor_ctype{i};
    file_bin_name = fullfile(out_dir, [num2str(i-1) '.txt']);
    file_bin_id = fopen(file_bin_name,'w');
    for row = 1:size(bin_info, 1)
        fprintf(file_bin_id, '%s %d %d %d %d %d %d\n', bin_info{row,1}, bin_info{row,2}, bin_info{row,3}, bin_info{row,4}, bin_info{row,5}, bin_info{row,6}, bin_info{row,7});
    end
    fclose(file_bin_id);
end
%delete(gcp);

end

end
