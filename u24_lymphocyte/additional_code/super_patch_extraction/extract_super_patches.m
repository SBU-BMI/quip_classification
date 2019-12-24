function extract_super_patches()

n_picked_each_bin = 5;
src_dir = 'super_patch_coords';
des_dir = 'super_patches';
cancertype_arr = {'blca' 'brca' 'cesc' 'coad' 'luad' 'lusc' 'paad' 'prad' 'read' 'skcm' 'stad' 'ucec' 'uvm'};

slide_folder_arr = {...
    '/data02/tcga_data/tumor/blca', ...
    '/data03/tcga_data/tumor/brca', ...
    '/data04/tcga_data/tumor/cesc', ...
    '/data07/tcga_data/tumor/coad', ...
    '/data01/tcga_data/tumor/luad', ...
    '/data02/tcga_data/tumor/lusc', ...
    '/data08/shared/lehhou/tcga/tumor/paad', ...
    '/data08/tcga_data/tumor/prad', ...
    '/data08/shared/lehhou/tcga/tumor/read', ...
    '/data01/tcga_data/tumor/skcm', ...
    '/data02/tcga_data/tumor/stad', ...
    '/data08/shared/lehhou/tcga/tumor/ucec', ...
    '/data08/tcga_data/tumor/uvm'};

parpool(12);
for i_cancer_type = 1:length(cancertype_arr)
    cancer_type = cancertype_arr{i_cancer_type};
    slide_folder = slide_folder_arr{i_cancer_type};
    disp(cancer_type);

    out_dir = fullfile(des_dir, cancer_type);
    if (exist(out_dir) == 0)
        mkdir(out_dir);
    end
    parfor i = 0:64
        out_dir_bin = fullfile(out_dir, num2str(i));
        if (exist(out_dir_bin) == 0)
            mkdir(out_dir_bin);
        end
        binfile = fullfile(src_dir, cancer_type, [num2str(i) '.txt']);
        fileid = fopen(binfile);
        patches = textscan(fileid, '%s%d%d%d%d%d%d', 'Delimiter',' ');
        names = patches{:,1};
        y1 = patches{:,2} - 1;
        x1 = patches{:,3} - 1;
        y2 = patches{:,4} - 1;
        x2 = patches{:,5} - 1;
        n1 = patches{:,6} - 1;
        n2 = patches{:,7} - 1;
        height = y2 - y1 + 1;
        width = x2 - x1 + 1;

        shuf = randperm(size(x1,1));
        n_pick = min(size(x1, 1), n_picked_each_bin);
        shuf = shuf(1:n_pick);

        for i_slide_idx = 1:n_pick
            i_slide = shuf(i_slide_idx);
            slidename = names{i_slide};

            % Get slidepath
            sl_list = dir([slide_folder, '/', slidename, '.*svs']);
            sl_list = {sl_list.name};
            if (length(sl_list) == 0)
                continue;
            end
            slidefilename = sl_list{1};
            slidepath = sprintf('%s/%s.svs', slide_folder, slidefilename);

            % Get mpp
            str = sprintf('/cm/shared/apps/extlibs/bin/openslide-show-properties %s | grep openslide.mpp-x',
                          slidepath);
            [status, mpp_line] = system(str);
            fields = strsplit(mpp_line, '''');
            mpp = str2num(fields{2});

            % Get patch_size
            pw_20X = 100;
            mag = 10.0 / mpp;
            patch_size = double(int32(floor(10 * pw_20X * mag / 20))) / 10.0;

            y1_slide = y1(i_slide, 1);
            x1_slide = x1(i_slide, 1);
            n1_slide = n1(i_slide, 1);
            n2_slide = n2(i_slide, 1);
            height_slide = height(i_slide, 1);
            width_slide  = width(i_slide, 1);
            real_y1_slide = y1_slide * patch_size;
            real_x1_slide = x1_slide * patch_size;
            real_height_slide = height_slide * patch_size;
            real_width_slide = width_slide * patch_size;

            if (exist(slidepath) == 0)
                continue;
            end

            outpath = fullfile(out_dir_bin, [slidename '_' num2str(real_x1_slide) '_' num2str(real_y1_slide) '_' num2str(n1_slide) '_' num2str(n2_slide) '.png']);
            cmdline = sprintf('/cm/shared/apps/extlibs/bin/openslide-write-png %s %d %d 0 %d %d %s', slidepath, real_x1_slide, real_y1_slide, real_width_slide, real_height_slide, outpath);
            system(cmdline);
        end
    end
end
delete(gcp);

end
