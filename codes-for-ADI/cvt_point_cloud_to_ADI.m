function adi_image = cvt_point_cloud_to_ADI(calib_path_tmpl,im_path_tmpl,velo_path_tmpl,out_adi_path_tmpl)
% CVT_POINT_CLOUD_TO_ADI 
%   convert point cloud to altitude difference image
%   results will be written to the path defined by out_adi_path_tmpl
%
% INPUTS:
%   calib_path_tmpl - file path template for calibration parameters;
%   im_path_tmpl - file path template for visual images;
%   velo_path_tmpl - file path template for velodyne 3D point cloud data;
%   out_adi_path_tmpl - file path template to write altitude difference image;
%
% Notice: 
%   This is a re-implementation of the computation of altitude difference
%   image as described in the paper "Progressive LiDAR Adaptation for Road
%   Detection", Zhe Chen, et al., JAS 2019. The original codes were lost,
%   thus there could be difference in details comparing to the procedure
%   described by the original paper.
%
% Copyright 2020 Zhe Chen.  [zhe.chen1-at-sydney.edu.au]
% Licensed under the Simplified BSD License [see bsd.txt]

i = 0;
while 1
    %% read data
    try
        im = imread(sprintf(im_path_tmpl, i));
        imh = size(im,1); imw = size(im,2);
        i = i + 1;
    catch
        break;
    end
    fprintf('processing frame %d\n',i);
    trans_param = loadCalibration(sprintf(calib_path_tmpl, i));
    
    pt_data = read_point_cloud(velo_path_tmpl, i);
    
    %% map point cloud to image plane
    [im_pt_map, pt_mask] = map_point_cloud_on_image(trans_param, pt_data, imh, imw);
    
    %% interpolate heights
    dr = zeros(21,21); % dictionary of distances
    for dy = 1:21
        for dx = 1:21
            dr(dy,dx) = 1 / max(sqrt((dy-11)^2 + (dx-11)^2),1);
        end
    end
    
    % we first interpolate a dense height map from sparse points that are 
    % mapped on the image
    sparse_h_map = im_pt_map(:,:,3); 
    dense_h_map = zeros(size(sparse_h_map));
    pt_mask_tmp = pt_mask;
    for y = 1:imh
        for x = 1:imw
            % find true heights from neighboring pixels within [x1,y1,x2,y2]
            x1 = max(x - 10, 1); x2 = min(x + 10, imw);
            y1 = max(y - 10, 1); y2 = min(y + 10, imh);
            h = sparse_h_map(y,x);
            if pt_mask(y,x) == 1 % already has a true height
                dense_h_map(y,x) = h;
                continue;
            else 
                % in neighboring pixels, we average true heights weighted
                % according to their distances to current pixel to obtain 
                % the interpolated height
                hs = 0; n = 0;
                for xx = x1:x2
                    for yy = y1:y2
                        h_ = sparse_h_map(yy,xx);
                        if pt_mask(yy,xx) == 0 || (xx == x && yy == y)
                            continue;
                        else
                            r = dr(yy-y+11, xx-x+11);
                            hs = hs + h_ * r;
                            n = n + r;
                        end
                    end
                end
                if n > 0
                    hs = hs / n;
                    pt_mask_tmp(y,x) = 1;
                else
                    hs = 0; % if neighboring pixels also lack true heights
                end
                dense_h_map(y,x) = hs;
            end
        end
    end
    pt_mask = pt_mask_tmp;
    
    %% compute altitude difference image
    adi_image = zeros(size(sparse_h_map));    
    for y = 1:imh
        for x = 1:imw
            if pt_mask(y,x) < 1
                continue;
            end
            % compute altitude difference within 3x3 window
            x1 = max(x - 3, 1); x2 = min(x + 3, imw);
            y1 = max(y - 3, 1); y2 = min(y + 3, imh);
            h = dense_h_map(y,x);

            ads = 0; n = 0;
            maxad = 0;
            for xx = x1:x2
                for yy = y1:y2
                    if pt_mask(yy,xx) < 1 % no dense height
                        continue;
                    end
                    h_ = dense_h_map(yy,xx);
                    dh = h_ - h; 
                    % altitude difference w.r.t. vertical distance
                    if xx == x, vg = 0;
                    else, vg = dh / (xx - x);
                    end
                    % altitude difference w.r.t. horizontal distance
                    if yy == y, hg = 0;
                    else, hg = dh / (yy - y); 
                    end
                    % overall altitude difference
                    ad = sqrt(vg^2 + hg^2);
                    
                    % average altitude difference within 3x3 window
                    ads = ads + ad; n = n + 1;
                    % record maximum altitude difference on the image
                    if ad > maxad
                        maxad = ad;
                    end
                end
            end
            if n > 0
                adi_image(y,x) = ads / n;
            end
        end
    end
    
    % normalize altitude difference image
    tmp_adi_image = adi_image(adi_image > 0);
    tmp_adi_image = tmp_adi_image - min(tmp_adi_image(:)); 
    tmp_adi_image = tmp_adi_image * 20;
    adi_image(adi_image > 0) = tmp_adi_image;
    % smooth the image
    adi_image = imgaussfilt(sqrt(adi_image),2);
    adi_image(adi_image > 1) = 1;

%     imshow(adi_image);
%     waitforbuttonpress;    
    imwrite(adi_image, sprintf(out_adi_path_tmpl,i));
end
end

function [im_pt_map, pt_mask] = map_point_cloud_on_image(trans_param, pt_data, imh, imw)
    pt_tmp = trans_param * pt_data';
    pt_on_img = pt_tmp(1:2,:) ./ repmat(pt_tmp(3,:), [2,1]);
    pt_on_img = round(pt_on_img');
    
    % remove out of image points
    idx = (pt_on_img(:,1) <= 0) | (pt_on_img(:,1) > imw) | ...
        (pt_on_img(:,2) <= 0) | (pt_on_img(:,2) > imh);
    pt_on_img(idx,:) = []; pt_data(idx,:) = [];
    
    im_pt_map = zeros(imh,imw,3);
    pt_mask = zeros(imh,imw); % record already mapped position
    
    for pi = 1:size(pt_on_img,1)
        imx = pt_on_img(pi,1); imy = pt_on_img(pi,2);
        if pt_mask(imy,imx) < 1 || norm(squeeze(pt_data(pi,1:3))) ...
            < norm(squeeze(im_pt_map(imy,imx,:)))
            im_pt_map(imy,imx, :) = pt_data(pi,1:3);
            pt_mask(imy,imx) = 1;
        end
    end
end

function pt_data = read_point_cloud(velo_path_tmpl, i)
    fid = fopen(sprintf(velo_path_tmpl,i),'rb');
    pt_data = fread(fid,[4 inf],'single')';
    fclose(fid);
    
    idx = pt_data(:,1)<5;
    pt_data(idx,:) = [];
    pt_data = squeeze(pt_data);
    pt_data(:,4) = 1;
end

function calib = loadCalibration(file)
    R_rect = zeros(4,4);
    T_vc = zeros(4,4);
    
    fid = fopen(file, 'r');
    P2 = readVariable(fid,'P2',3,4);
    
    R_rect(1:3,1:3) = readVariable(fid,'R0_rect',3,3);
    R_rect(4,4) = 1;
    
    T_vc(1:3,:) = readVariable(fid,'Tr_velo_to_cam',3,4);
    T_vc(4,4) = 1;
    
    calib = P2 * R_rect * T_vc;

    fclose(fid);
end

function A = readVariable(fid,name,M,N)
    % rewind
    fseek(fid,0,'bof');

    % search for variable identifier
    success = 1;
    while success>0
      [str,success] = fscanf(fid,'%s',1);
      if strcmp(str,[name ':'])
        break;
      end
    end

    % return if variable identifier not found
    if ~success
      A = [];
      return;
    end

    % fill matrix
    A = zeros(M,N);
    for m=1:M
      for n=1:N
        [val,success] = fscanf(fid,'%f',1);
        if success
          A(m,n) = val;
        else
          A = [];
          return;
        end
      end
    end
end