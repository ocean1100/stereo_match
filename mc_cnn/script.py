# @Author: Hao G <hao>
# @Date:   2018-01-05T14:24:09+00:00
# @Email:  hao.guan@digitalbridge.eu
# @Last modified by:   hao
# @Last modified time: 2018-01-10T15:15:12+00:00



./main.lua kitti fast -a predict -net_fname net/net_kitti_fast_-a_train_all.t7 -left /home/hao/MyCode/disparity_estimation/imgs/aloeL.jpg -right /home/hao/MyCode/disparity_estimation/imgs/aloeR.jpg -disp_max 228 (-sm_terminate cnn)
./main.lua mb slow -a predict -net_fname net/net_mb_slow_-a_train_all.t7 -left /home/hao/MyCode/disparity_estimation/imgs/rectified_l.png -right /home/hao/MyCode/disparity_estimation/imgs/rectified_r.png -disp_max 228 (-sm_terminate cnn)
luajit samples/bin2png_1280.lua
