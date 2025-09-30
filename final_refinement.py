#!/usr/bin/env python3
"""
fast_eval_seg_and_surf.py

Fast evaluation (Dice + ASSD) using:
 - Dice: remap predicted classes -> GT label codes (majority overlap), then SimpleITK LabelOverlap.
 - ASSD: extract surfaces via skimage.measure.marching_cubes_lewiner for each label (GT and mapped-PRED),
         convert to vtk PolyData, compute distances via vtkDistancePolyDataFilter.

Usage (example):
python tools/fast_eval_seg_and_surf.py \
  --pred "E:\Capstone\Trial-2\MeshDeformNet\data\mmwhs_outputs\ct_train_1001_image.nii.gz" \
  --gt   "E:\Capstone\Trial-2\MeshDeformNet\data\mmwhs\ct_test_seg\ct_train_1001_label.nii" \
  --labels 205 420 500 550 600 820 850 \
  --out results.txt
"""
import os, sys, argparse
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import distance_transform_edt
from skimage import measure
import vtk
from vtk.util.numpy_support import numpy_to_vtk

def auto_create_label_map(pred_img, gt_img, exclude_zero=True, verbose=True):
    # pred_img, gt_img: SimpleITK.Image already resampled to same grid
    pnp = sitk.GetArrayFromImage(pred_img).astype(int)
    gnp = sitk.GetArrayFromImage(gt_img).astype(int)
    pvals, _ = np.unique(pnp, return_counts=True)
    mapping = {}
    if verbose:
        print("Computing mapping from predicted classes -> GT labels by majority overlap")
        print("Pred unique:", pvals.tolist())
    for p in pvals:
        p = int(p)
        if exclude_zero and p == 0:
            continue
        mask = (pnp == p)
        if mask.sum() == 0:
            if verbose:
                print("Pred class %d: empty -> skip" % p)
            continue
        gv, cnt = np.unique(gnp[mask], return_counts=True)
        order = np.argsort(-cnt)
        chosen = None
        for idx in order:
            cand = int(gv[idx])
            if cand != 0:
                chosen = cand
                break
        if chosen is None:
            chosen = int(gv[order[0]])
        mapping[p] = chosen
        if verbose:
            print("  %d -> %d  counts: %s" % (p, chosen, str(dict(zip(gv.tolist(), cnt.tolist())))))
    return mapping

def remap_pred_to_gt_codes(pred_img, mapping, default_other=0):
    arr = sitk.GetArrayFromImage(pred_img).astype(int)
    out = np.zeros_like(arr, dtype=np.int32)
    for p, g in mapping.items():
        out[arr == p] = int(g)
    # keep zeros as zero, unmapped nonzeros set to default_other (0)
    img = sitk.GetImageFromArray(out.astype(np.int16))
    img.SetOrigin(pred_img.GetOrigin())
    img.SetSpacing(pred_img.GetSpacing())
    img.SetDirection(pred_img.GetDirection())
    return img

def compute_dice_per_label(pred_img_path, gt_img_path, labels, verbose=True):
    pred = sitk.ReadImage(pred_img_path)
    gt = sitk.ReadImage(gt_img_path)
    # resample pred -> gt geometry
    pred_res = sitk.Resample(pred, gt, sitk.Transform(), sitk.sitkNearestNeighbor, 0, pred.GetPixelID())
    # create label overlap filter
    mom = sitk.LabelOverlapMeasuresImageFilter()
    res = {}
    for lbl in labels:
        if lbl == 0: continue
        pbin = sitk.BinaryThreshold(pred_res, lbl, lbl, 1, 0)
        gbin = sitk.BinaryThreshold(gt, lbl, lbl, 1, 0)
        mom.Execute(gbin, pbin)
        res[int(lbl)] = float(mom.GetDiceCoefficient())
        if verbose:
            print("Dice label %d: %g" % (lbl, res[int(lbl)]))
    return res

def marching_to_vtk(verts, faces, origin, spacing, direction):
    # verts from skimage are in (z,y,x) voxel coordinates (floating)
    # convert to (ix,iy,iz) = (x,y,z) then to physical: phys = direction @ (idx * spacing).T + origin
    # verts shape (N,3) in (z,y,x)
    idx = verts[:, [2,1,0]]
    phys = (np.array(direction).reshape(3,3) @ (idx * spacing).T).T + np.array(origin)
    # build vtk polydata
    vtk_pts = vtk.vtkPoints()
    for p in phys:
        vtk_pts.InsertNextPoint(float(p[0]), float(p[1]), float(p[2]))
    polys = vtk.vtkCellArray()
    for f in faces:
        polys.InsertNextCell(3)
        polys.InsertCellPoint(int(f[0])); polys.InsertCellPoint(int(f[1])); polys.InsertCellPoint(int(f[2]))
    pd = vtk.vtkPolyData()
    pd.SetPoints(vtk_pts)
    pd.SetPolys(polys)
    return pd

def surf_from_label_array(arr, label, gt_sitk_img):
    # arr is numpy (z,y,x)
    mask = (arr == label).astype(np.uint8)
    if mask.sum() == 0:
        return None
    # skimage marching: runs on volume with coordinates as array indices (z,y,x)
    # marching_cubes_lewiner returns verts (N,3) in (z,y,x) coordinates, faces (M,3)
    verts, faces, _, _ = measure.marching_cubes_lewiner(mask, level=0.5)
    # convert verts/faces to vtk polydata using physical transform from gt image
    origin = np.array(gt_sitk_img.GetOrigin(), dtype=float)
    spacing = np.array(gt_sitk_img.GetSpacing(), dtype=float)
    direction = np.array(gt_sitk_img.GetDirection(), dtype=float)
    pd = marching_to_vtk(verts, faces, origin, spacing, direction)
    return pd

def compute_assd_between_polydatas(pred_pd, gt_pd):
    # returns symmetric average distance (ASSD)
    if pred_pd is None or gt_pd is None:
        return float('nan')
    # pred->gt
    df = vtk.vtkDistancePolyDataFilter()
    df.SetInputData(0, pred_pd)
    df.SetInputData(1, gt_pd)
    df.SignedDistanceOff()
    df.Update()
    arr0 = vtk.util.numpy_support.vtk_to_numpy(df.GetOutput().GetPointData().GetArray(0))
    if arr0.size == 0:
        return float('nan')
    d_p2g = arr0.mean()
    # gt->pred
    df2 = vtk.vtkDistancePolyDataFilter()
    df2.SetInputData(0, gt_pd)
    df2.SetInputData(1, pred_pd)
    df2.SignedDistanceOff()
    df2.Update()
    arr1 = vtk.util.numpy_support.vtk_to_numpy(df2.GetOutput().GetPointData().GetArray(0))
    if arr1.size == 0:
        return float('nan')
    d_g2p = arr1.mean()
    return 0.5 * (d_p2g + d_g2p)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--gt", required=True)
    p.add_argument("--labels", nargs='+', type=int, default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--verbose", action='store_true')
    args = p.parse_args()

    pred_fn = args.pred
    gt_fn = args.gt
    pred_img = sitk.ReadImage(pred_fn)
    gt_img = sitk.ReadImage(gt_fn)
    # resample pred to gt geometry for consistent overlap/mapping
    pred_res = sitk.Resample(pred_img, gt_img, sitk.Transform(), sitk.sitkNearestNeighbor, 0, pred_img.GetPixelID())

    mapping = auto_create_label_map(pred_res, gt_img, exclude_zero=True, verbose=args.verbose)
    if args.verbose:
        print("Mapping:", mapping)

    # remap predicted to GT codes
    pred_mapped = remap_pred_to_gt_codes(pred_res, mapping)
    # write remapped short preview (optional)
    if args.out:
        tmp_remap = os.path.splitext(args.out)[0] + "_pred_mapped.nii.gz"
        sitk.WriteImage(pred_mapped, tmp_remap)
        if args.verbose:
            print("Wrote remapped pred to", tmp_remap)

    # Dice per label
    labels = args.labels if args.labels is not None else sorted([int(x) for x in np.unique(sitk.GetArrayFromImage(gt_img)) if x!=0])
    if args.verbose:
        print("Labels to evaluate:", labels)
    # write pred_mapped to temp on disk for SimpleITK overlap filter
    tmp_pred_path = os.path.join(os.path.dirname(pred_fn), "__tmp_pred_mapped.nii.gz")
    sitk.WriteImage(pred_mapped, tmp_pred_path)
    dice = compute_dice_per_label(tmp_pred_path, gt_fn, labels, verbose=args.verbose)

    # ASSD: compute surfaces using marching cubes on both volumes
    pred_arr = sitk.GetArrayFromImage(pred_mapped).astype(int)  # z,y,x
    gt_arr = sitk.GetArrayFromImage(gt_img).astype(int)
    assd = {}
    for lbl in labels:
        if lbl == 0:
            continue
        if args.verbose:
            print("Extracting surfaces for label", lbl)
        pred_pd = surf_from_label_array(pred_arr, lbl, gt_img)
        gt_pd   = surf_from_label_array(gt_arr,  lbl, gt_img)
        d = compute_assd_between_polydatas(pred_pd, gt_pd)
        assd[int(lbl)] = float(d) if d is not None else float('nan')
        if args.verbose:
            print(" ASSD label %d: %s" % (lbl, str(assd[int(lbl)])))

    # print and optionally save
    out = {"dice": dice, "assd": assd, "mapping": mapping}
    if args.out:
        import json
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print("Wrote results to", args.out)
    print("Dice:", dice)
    print("ASSD:", assd)
    print("Mapping:", mapping)

if __name__ == "__main__":
    main()
 
