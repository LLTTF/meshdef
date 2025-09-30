# meshdef

python tools\fast_eval_seg_and_surf.py ^
  --pred "E:\Capstone\Trial-2\MeshDeformNet\data\mmwhs_outputs\ct_train_1001_image.nii.gz" ^
  --gt "E:\Capstone\Trial-2\MeshDeformNet\data\mmwhs\ct_test_seg\ct_train_1001_label.nii" ^
  --labels 205 420 500 550 600 820 850 ^
  --out "E:\Capstone\Trial-2\MeshDeformNet\data\mmwhs_outputs\ct_train_1001_eval.json" ^
  --verbose
