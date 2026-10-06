#%%
import numpy as np 
import nibabel as nib
from pathlib import Path
import os 
from scipy.ndimage import binary_fill_holes
from nibabel.processing import resample_from_to

def axial_distribution(img):
    img = nib.funcs.as_closest_canonical(img)
    img_y = img.get_fdata().sum(axis=(0,1)) * np.prod(img.header.get_zooms()[:2]) / 1000
    nz = img.shape[-1]
    img_x = img.affine @ np.array([np.zeros(nz),np.zeros(nz),np.arange(nz),np.ones(nz)])
    img_x = img_x[-2,:]
    return img_x, img_y

def axial_distribution_mask(img, num_classes=13):
    dataobj = np.asanyarray(img.dataobj)
    nz = img.shape[-1]
    img_y = np.zeros((nz,num_classes))
    for ix in range(1,num_classes+1):
        img_y[:,ix-1] = (dataobj==ix).sum(axis=(0,1))
    img_x = img.affine @ np.array([np.zeros(nz),np.zeros(nz),np.arange(nz),np.ones(nz)])
    img_x = img_x[-2,:]
    img_y*= np.prod(img.header.get_zooms()[:2]) / 1000
    return img_x, img_y

def get_ct_cylindrical_mask(ct_image):
    mask = ct_image.get_fdata() > -1023.5
    mask = mask.any(axis=2,keepdims=True)
    mask = binary_fill_holes(mask)
    mask = np.repeat(mask,ct_image.shape[2],axis=2)
    return nib.Nifti1Image(mask.astype("uint8"),affine=ct_image.affine)

def crop_pet_activity_to_ct_axial_fov(pet_img,ct_image):
    mask = get_ct_cylindrical_mask(ct_image)
    mask = resample_from_to(mask,pet_img,order=0)
    pet_arr = pet_img.get_fdata().copy()
    pet_arr[mask.get_fdata()==0] = 0
    return nib.Nifti1Image(pet_arr,pet_img.affine)
    
def make_distribution_functions(pet_img, ts_total_img, ts_body_img, ts_tissue_img,ct_img, out_directory=None):
    if out_directory is not None:
        os.makedirs(out_directory,exist_ok=True)
    if not isinstance(out_directory,Path) and out_directory is not None:
        out_directory = Path(out_directory)
    data = {}

    x,y = axial_distribution_mask(ts_total_img,num_classes=117)
    data["ts_total_x"] = x
    data["ts_total_y"] = y

    pet_img = crop_pet_activity_to_ct_axial_fov(pet_img,ct_img)
    x,y = axial_distribution(pet_img)
    data["pet_x"] = x
    data["pet_y"] = y

    x,y = axial_distribution_mask(ts_body_img,num_classes=2)
    data["ts_body_x"] = x
    data["ts_body_y"] = y

    if ts_tissue_img is not None:
        x,y = axial_distribution_mask(ts_tissue_img,num_classes=3)
        data["ts_tissues_x"] = x
        data["ts_tissues_y"] = y

    
    return data 

