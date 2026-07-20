import os
import numpy as np
from os.path import join
import SimpleITK as sitk
from skimage import transform


def NiiDataRead(path, image_only=False, as_type=np.float32):
    """
    Read a NIfTI medical image and return its data, spacing, origin, and direction.

    Args:
        path (str): Path to the NIfTI file.
        as_type (np.dtype): Desired data type of the returned image array.

    Returns:
        volumn (ndarray): 3D image data in [z, y, x] order.
        spacing_ (ndarray): Voxel spacing in [z, y, x] order.
        origin (tuple): Image origin in physical space.
        direction (tuple): Image orientation (direction cosine matrix).
    """
    nii = sitk.ReadImage(path)
    spacing = nii.GetSpacing()  # [x,y,z]
    volumn = sitk.GetArrayFromImage(nii)  # [z,y,x]
    if image_only:
        return volumn.astype(as_type)
    else:
        origin = nii.GetOrigin()
        direction = nii.GetDirection()

        spacing_x = spacing[0]
        spacing_y = spacing[1]
        spacing_z = spacing[2]

        spacing_ = np.array([spacing_z, spacing_y, spacing_x])
        return volumn.astype(as_type), spacing_.astype(np.float32), origin, direction


def NiiDataWrite(save_path, volumn, spacing=np.array([1,1,1]), origin=(0, 0, 0), direction=[1, 0, 0, 0, 1, 0, 0, 0, 1], as_type=np.float32):
    """
    Save a 3D numpy array as a NIfTI file with spatial information.

    Args:
        save_path (str): Destination path to save the NIfTI file.
        volumn (ndarray): 3D image data [z, y, x].
        spacing (ndarray): Spacing in [z, y, x] order.
        origin (tuple): Image origin.
        direction (tuple): Image orientation.
        as_type (np.dtype): Data type to save as.
    """
    spacing = spacing.astype(np.float64)
    raw = sitk.GetImageFromArray(volumn[:, :, :].astype(as_type))

    # Convert spacing back to [x, y, z] for SimpleITK
    spacing_ = (spacing[2], spacing[1], spacing[0])
    raw.SetSpacing(spacing_)
    raw.SetOrigin(origin)
    raw.SetDirection(direction)
    sitk.WriteImage(raw, save_path)


if __name__ == '__main__':
    taget_size = (32, 160, 192)
    values_clip = (-55, 145)
    data_root = './RAW_DATA/'
    save_root = './Processed_DATA/'

    for dataset in ['Train', 'Val']:
        for ID in os.listdir(join(data_root, dataset)):
            print(ID)
            for modal in ['NC', 'AP', 'PVP', 'DP']:
                img, spacing, origin, direction = NiiDataRead(join(data_root, dataset, ID, f"{modal}.nii.gz"))
                mask_liver, _, _, _ = NiiDataRead(join(data_root, dataset, ID, 'Liver_mask.nii.gz'))
                mask_tumor, _, _, _ = NiiDataRead(join(data_root, dataset, ID, 'Tumor_mask.nii.gz'))

                z_, x_, y_ = mask_liver.nonzero()
                z1 = z_.min()
                z2 = z_.max()
                x1 = x_.min()
                x2 = x_.max()
                y1 = y_.min()
                y2 = y_.max()

                img = img[z1: z2 + 1, x1: x2 + 1, y1: y2 + 1]
                mask_liver = mask_liver[z1: z2 + 1, x1: x2 + 1, y1: y2 + 1]
                mask_tumor = mask_tumor[z1: z2 + 1, x1: x2 + 1, y1: y2 + 1]

                img = np.clip(img, values_clip[0], values_clip[1])
                img = (img - values_clip[0]) / (values_clip[1] - values_clip[0]) * 2 - 1

                # 重新计算 spacing
                spacing_z = (spacing[0] * img.shape[0]) / taget_size[0]
                spacing_x = (spacing[1] * img.shape[1]) / taget_size[1]
                spacing_y = (spacing[2] * img.shape[2]) / taget_size[2]
                spacing = np.array([spacing_z, spacing_x, spacing_y])


                img = transform.resize(img, taget_size, order=0, mode='constant', clip=False, preserve_range=True, anti_aliasing=False)
                mask_liver = transform.resize(mask_liver, taget_size, order=0, mode='constant', clip=False, preserve_range=True, anti_aliasing=False)
                mask_tumor = transform.resize(mask_tumor, taget_size, order=0, mode='constant', clip=False, preserve_range=True, anti_aliasing=False)

                img_liver = np.copy(img)
                img_tumor = np.copy(img)
                img_liver[mask_liver == 0] = -1
                img_tumor[mask_tumor == 0] = -1
                os.makedirs(join(save_root, dataset, ID), exist_ok=True)
                NiiDataWrite(join(save_root, dataset, ID, f'{modal}_liver.nii.gz'), img_liver, spacing, origin, direction)
                NiiDataWrite(join(save_root, dataset, ID, f'{modal}_tumor.nii.gz'), img_tumor, spacing, origin, direction)