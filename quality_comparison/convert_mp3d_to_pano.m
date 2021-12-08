% Copyright (c) Facebook, Inc. and its affiliates.
% This source code is licensed under the MIT license found in the
% LICENSE file in the root directory of this source tree.

pano_basic_root = "<UPDATE PANO BASIC PATH>";
mp3d_skybox_root = "<PATH TO SKYBOXES>";
mp3d_pano_save_root = "<PATH TO SAVE PANORAMAS>";


addpath(pano_basic_root + '/' + 'CoordsTransfrom');
addpath(pano_basic_root + '/' + 'icosahedron2sphere');
addpath(pano_basic_root + '/' + 'Projection');
addpath(genpath(pano_basic_root + '/' + 'BasicProcessing'));
addpath(pano_basic_root + '/' + 'visualization');


intrinsics = [1280 1024 1072.27 1073.08 626.949 506.267 0 0 0 0 0];

vx = [-pi/2 -pi/2 0 pi/2 pi -pi/2];
vy = [pi/2 0 0 0 0 -pi/2];


T = readtable('data/mp3d_data_paths.csv', 'Delimiter', ',');

steps = length(T.Path);
parfor (i = 1:steps, 16)
    uid = T.Path{i};
    sepImg = [];
    for a = 1:6
        img_path = sprintf(mp3d_skybox_root + '/%s_skybox%d_sami.jpg', uid, a-1);
        sepImg(a).img = im2double(imread(img_path));
        sepImg(a).vx = vx(a);
        sepImg(a).vy = vy(a);
        sepImg(a).fov = pi/2+0.001;
        sepImg(a).sz = size(sepImg(a).img);
    end
    panoskybox = combineViews( sepImg, 2048, 1024 );
    imwrite(panoskybox, sprintf(mp3d_pano_save_root + '/%s_pano.jpg', uid));
end
