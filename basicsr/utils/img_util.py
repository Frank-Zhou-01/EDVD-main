import cv2
import math
import numpy as np
import os
import torch
from torchvision.utils import make_grid


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.

    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.

    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            if img.dtype == 'float64':
                img = img.astype('float32')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    if isinstance(imgs, list):
        return [_totensor(img, bgr2rgb, float32) for img in imgs]
    else:
        return _totensor(imgs, bgr2rgb, float32)


def voxel_norm(voxel):
    """
    Norm the voxel

    :param voxel: The unnormed voxel grid
    :return voxel: The normed voxel grid
    """
    nonzero_ev = (voxel != 0)
    num_nonzeros = nonzero_ev.sum()
    # print('DEBUG: num_nonzeros:{}'.format(num_nonzeros))
    if num_nonzeros > 0:
        # compute mean and stddev of the **nonzero** elements of the event tensor
        # we do not use PyTorch's default mean() and std() functions since it's faster
        # to compute it by hand than applying those funcs to a masked array
        mean = voxel.sum() / num_nonzeros
        stddev = torch.sqrt((voxel ** 2).sum() / num_nonzeros - mean ** 2)
        mask = nonzero_ev.float()
        voxel = mask * (voxel - mean) / stddev

    return voxel


import numpy as np


def crop_image_center_numpy(image, crop_size):
    """
    从中心裁剪图片到指定大小。

    参数:
        image (numpy.ndarray): 输入图片，形状为 (height, width, channels)。
        crop_size (tuple): 裁剪后的图片大小，格式为 (crop_height, crop_width)。

    返回:
        numpy.ndarray: 裁剪后的图片。
    """
    # 获取图片的高度、宽度和通道数
    _, _, img_height, img_width = image.shape

    # 获取裁剪的高度和宽度
    crop_height, crop_width = crop_size

    # 检查裁剪尺寸是否小于图片尺寸
    if crop_height > img_height or crop_width > img_width:
        raise ValueError("裁剪尺寸大于图片尺寸，请检查输入参数！")

    # 计算裁剪的起始点（确保从中心裁剪）
    start_y = (img_height - crop_height) // 2
    start_x = (img_width - crop_width) // 2

    # 使用 numpy 的索引操作裁剪图片
    cropped_image = image[:, :, start_y:start_y + crop_height, start_x:start_x + crop_width]

    return cropped_image


def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(_tensor, nrow=int(math.sqrt(_tensor.size(0))), normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. ' f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def tensor2img_fast(tensor, rgb2bgr=True, min_max=(0, 1)):
    """This implementation is slightly faster than tensor2img.
    It now only supports torch tensor with shape (1, c, h, w).

    Args:
        tensor (Tensor): Now only support torch tensor with (1, c, h, w).
        rgb2bgr (bool): Whether to change rgb to bgr. Default: True.
        min_max (tuple[int]): min and max values for clamp.
    """
    output = tensor.squeeze(0).detach().clamp_(*min_max).permute(1, 2, 0)
    output = (output - min_max[0]) / (min_max[1] - min_max[0]) * 255
    output = output.type(torch.uint8).cpu().numpy()
    if rgb2bgr:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    return output


def imfrombytes(content, flag='color', float32=False):
    """Read an image from bytes.

    Args:
        content (bytes): Image bytes got from files or other streams.
        flag (str): Flags specifying the color type of a loaded image,
            candidates are `color`, `grayscale` and `unchanged`.
        float32 (bool): Whether to change to float32., If True, will also norm
            to [0, 1]. Default: False.

    Returns:
        ndarray: Loaded image array.
    """
    img_np = np.frombuffer(content, np.uint8)
    imread_flags = {'color': cv2.IMREAD_COLOR, 'grayscale': cv2.IMREAD_GRAYSCALE, 'unchanged': cv2.IMREAD_UNCHANGED}
    img = cv2.imdecode(img_np, imread_flags[flag])
    if float32:
        img = img.astype(np.float32) / 255.
    return img


def imwrite(img, file_path, params=None, auto_mkdir=True, data_augment=None):
    """Write image to file.

    Args:
        data_augment:
        img (ndarray): Image array to be written.
        file_path (str): Image file path.
        params (None or list): Same as opencv's :func:`imwrite` interface.
        auto_mkdir (bool): If the parent folder of `file_path` does not exist,
            whether to create it automatically.

    Returns:
        bool: Successful or not.
    """
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)

    # 几何变换
    data_augment = str(data_augment)
    if data_augment == '90':
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if data_augment == '180':
        img = cv2.rotate(img, cv2.ROTATE_180)
    if data_augment == '270':
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if data_augment == 'V':
        img = cv2.flip(img, 0)
    if data_augment == 'H':
        img = cv2.flip(img, 1)
    if data_augment == 'T':
        img = cv2.transpose(img)

    return cv2.imwrite(file_path, img, params)


def image_augment(imgs, augment='', dataset=''):

    if dataset == 'GOPRO' or dataset == 'REVD':
        imgs = [np.transpose(img, (1, 2, 0)) for img in imgs]

    data_augment = str(augment)
    if data_augment == '90':
        imgs = [cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) for img in imgs]
    if data_augment == '180':
        imgs = [cv2.rotate(img, cv2.ROTATE_180) for img in imgs]
    if data_augment == '270':
        imgs = [cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE) for img in imgs]
    if data_augment == 'V':
        imgs = [cv2.flip(img, 0) for img in imgs]
    if data_augment == 'H':
        imgs = [cv2.flip(img, 1) for img in imgs]
    if data_augment == 'T':
        imgs = [cv2.transpose(img) for img in imgs]

    if dataset == 'GOPRO' or dataset == 'REVD':
        imgs = [np.transpose(img, (2, 0, 1)) for img in imgs]

    return imgs


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [v[crop_border:-crop_border, crop_border:-crop_border, ...] for v in imgs]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border, ...]
