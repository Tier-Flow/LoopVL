# Adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py.
# Below is the original copyright:
# Copyright 2024 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Image processor class for PenguinVL."""

import math
from typing import Dict, List, Optional, Union

import numpy as np

import torch
from transformers.image_processing_utils import BaseImageProcessor, BatchFeature
from transformers.image_utils import ImageInput
from transformers.image_transforms import (
    convert_to_rgb,
    resize,
    to_channel_dimension_format,
)
from transformers.image_utils import (
    OPENAI_CLIP_MEAN,
    OPENAI_CLIP_STD,
    ChannelDimension,
    ImageInput,
    PILImageResampling,
    get_image_size,
    infer_channel_dimension_format,
    is_scaled_image,
    is_valid_image,
    make_list_of_images,
    to_numpy_array,
)
try:
    from transformers.image_utils import VideoInput
except:
    from transformers.video_utils import VideoInput
from transformers.utils import TensorType, is_vision_available, logging


logger = logging.get_logger(__name__)


if is_vision_available():
    from PIL import Image


def is_valid_video(video) -> bool:
    if isinstance(video, (list, tuple)):
        return all(is_valid_image(frame) for frame in video)
    elif isinstance(video, np.ndarray):
        return video.ndim == 4
    elif isinstance(video, torch.Tensor):
        return video.ndim == 4
    return False


def make_batched_images(images) -> List[List[ImageInput]]:
    """
    Normalize visual inputs to ``List[List[ImageInput]]`` – a list of *clips*,
    where each clip is a list of frames.

    Supported input formats::

        Nested clips : [[image], [f1, f2, ...], ...]   → returned as-is
        Flat frames  : [f1, f2, ...]                    → [[f1, f2, ...]]
        Single image : image                            → [[image]]

    Returns:
        List of clips, where each clip is a list of valid images / frames.
    """
    if isinstance(images, (list, tuple)) and len(images) > 0:
        if isinstance(images[0], (list, tuple)):
            return [list(clip) for clip in images]
        if all(is_valid_image(f) for f in images):
            return [list(images)]
    if is_valid_image(images):
        return [[images]]
    raise ValueError(f"Could not make batched images from {images}")


def simple_batched_resize(
    images, 
    factor: int = 28, 
    min_tokens: int = 4 * 4, 
    max_tokens: int = 16384, 
    input_data_format: str = None, 
    frame_types=None
):
    """
    Compute per-frame target (h, w) for a video frame list under a token budget (key/intermediate may differ).

    Uses the Temporal Redundancy-Aware (TRA) token compression strategy: key and intermediate frames
    can have different target areas (e.g. 1:16 ratio when compressing) to stay within max_tokens.

    Args:
        images: List of video frames (each PIL Image or ndarray).
        factor: Alignment granularity (height and width are multiples of factor), default 28.
        min_tokens: Minimum tokens per frame (used to derive min_pixels), default 16.
        max_tokens: Token cap for total pixel budget, default 16384.
        input_data_format: Channel format when not PIL, e.g. "channels_first".
        frame_types: Per-frame type list, 0=key, 1=intermediate; None means all key.

    Returns:
        image_sizes: List of (h, w) per frame, one-to-one with images.
    """
    min_pixels = min_tokens * factor * factor * 1.5
    max_pixels = max_tokens * factor * factor * 0.95

    # --- Base info ---
    first_image = images[0]
    if isinstance(first_image, Image.Image):
        width, height = first_image.size
    else:
        height, width = get_image_size(first_image, channel_dim=input_data_format)

    aspect_ratio = height / width
    raw_area = height * width

    num_frames = len(images)
    if frame_types is not None:
        ft_list = frame_types.tolist() if hasattr(frame_types, 'tolist') else frame_types
        num_intermediate = ft_list.count(1)
        num_key = ft_list.count(0)
    else:
        num_key = num_frames
        num_intermediate = 0
        ft_list = [0] * num_frames

    def get_dims_from_area(target_area, ar, fac):
        """Compute aligned (h, w) from target area and aspect ratio; area = w²·ar => w = sqrt(area/ar)."""
        w_new = math.sqrt(target_area / ar)
        h_new = w_new * ar

        h_bar = round(h_new / fac) * fac
        w_bar = round(w_new / fac) * fac
        h_bar = max(h_bar, fac)
        w_bar = max(w_bar, fac)

        return h_bar, w_bar

    # --- Stage 1: No-downscale check ---
    # If total pixels within budget, keep original size for both key and intermediate frames.
    total_raw_pixels = num_frames * raw_area
    target_key_area = raw_area
    target_intermediate_area = raw_area

    if total_raw_pixels > max_pixels:
        # --- Stage 2: Sync compression ---
        # Over budget: compress with 1:16 area ratio, intermediate_area = key_area / 16.
        # Constraint: N_key·A_key + N_intermediate·(A_key/16) = max_pixels => A_key = max_pixels / (N_key + N_intermediate/16).
        effective_count = num_key + (num_intermediate / 16.0)
        calc_key_area = max_pixels / effective_count
        calc_intermediate_area = calc_key_area / 16.0

        # --- Stage 3: Intermediate-frame floor ---
        # If computed intermediate area is below min_pixels, pin intermediate to min_pixels and give remaining budget to key.
        if calc_intermediate_area >= min_pixels:
            target_key_area = calc_key_area
            target_intermediate_area = calc_intermediate_area
        else:
            target_intermediate_area = min_pixels
            pixels_taken_by_intermediate = num_intermediate * min_pixels
            remaining_for_key = max_pixels - pixels_taken_by_intermediate
            target_key_area = remaining_for_key / num_key

        # --- Stage 4: Key-frame hard floor ---
        if target_key_area < min_pixels:
            target_key_area = min_pixels

    # --- Area to aligned dimensions ---
    k_h, k_w = get_dims_from_area(target_key_area, aspect_ratio, factor)
    if num_intermediate > 0:
        i_h, i_w = get_dims_from_area(target_intermediate_area, aspect_ratio, factor)
    else:
        i_h, i_w = 0, 0

    def ensure_min_hw(h, w, min_p, raw_ar):
        """If area still below min_pixels after alignment (rounding), recompute from min area and align upward."""
        if h * w < min_p:
            w = math.sqrt(min_p / raw_ar)
            h = w * raw_ar
            h = math.ceil(h / factor) * factor
            w = math.ceil(w / factor) * factor
        return h, w

    k_h, k_w = ensure_min_hw(k_h, k_w, min_pixels, aspect_ratio)
    if num_intermediate > 0:
        i_h, i_w = ensure_min_hw(i_h, i_w, min_pixels, aspect_ratio)

    image_sizes = [
        (i_h, i_w) if ft_list[i] == 1 else (k_h, k_w)
        for i in range(num_frames)
    ]
    return image_sizes


class PenguinVLImageProcessor(BaseImageProcessor):
    r"""
    Constructs a PenguinVL image processor that dynamically resizes images based on the original images.

    Args:
        do_resize (`bool`, *optional*, defaults to `True`):
            Whether to resize the image's (height, width) dimensions.
        resample (`PILImageResampling`, *optional*, defaults to `Resampling.BICUBIC`):
            Resampling filter to use when resizing the image.
        do_rescale (`bool`, *optional*, defaults to `True`):
            Whether to rescale the image by the specified scale `rescale_factor`.
        rescale_factor (`int` or `float`, *optional*, defaults to `1/255`):
            Scale factor to use if rescaling the image.
        do_normalize (`bool`, *optional*, defaults to `True`):
            Whether to normalize the image.
        image_mean (`float` or `List[float]`, *optional*, defaults to `[0.48145466, 0.4578275, 0.40821073]`):
            Mean to use if normalizing the image. This is a float or list of floats for each channel in the image.
        image_std (`float` or `List[float]`, *optional*, defaults to `[0.26862954, 0.26130258, 0.27577711]`):
            Standard deviation to use if normalizing the image. This is a float or list of floats for each channel in the image.
        do_convert_rgb (`bool`, *optional*, defaults to `True`):
            Whether to convert the image to RGB.
        min_pixels (`int`, *optional*, defaults to `56 * 56`):
            The min pixels of the image to resize the image.
        max_pixels (`int`, *optional*, defaults to `28 * 28 * 1280`):
            The max pixels of the image to resize the image.
        patch_size (`int`, *optional*, defaults to 14):
            The spacial patch size of the vision encoder.
    """

    model_input_names = ["pixel_values", "grid_sizes", "merge_sizes"]

    def __init__(
        self,
        do_resize: bool = True,
        resample: PILImageResampling = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: Union[int, float] = 1 / 255,
        do_normalize: bool = True,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = True,
        min_tokens: int = 4 * 4,
        max_tokens: int = 16384,
        patch_size: int = 14,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.do_resize = do_resize
        self.resample = resample
        self.do_rescale = do_rescale
        self.rescale_factor = rescale_factor
        self.do_normalize = do_normalize
        self.image_mean = image_mean if image_mean is not None else OPENAI_CLIP_MEAN
        self.image_std = image_std if image_std is not None else OPENAI_CLIP_STD
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.patch_size = patch_size
        self.do_convert_rgb = do_convert_rgb

    def _allocate_token_budget(self, clips, clip_merge_sizes, input_data_format):
        """Distribute self.max_tokens across clips proportionally to their raw token counts."""
        clip_raw_tokens = []
        for clip, ms in zip(clips, clip_merge_sizes):
            first_frame = clip[0]
            if isinstance(first_frame, Image.Image):
                w, h = first_frame.size
            else:
                h, w = get_image_size(first_frame, channel_dim=input_data_format)
            factor = self.patch_size * ms
            clip_raw_tokens.append(len(clip) * h * w / (factor * factor))

        total_raw_tokens = sum(clip_raw_tokens)
        if total_raw_tokens <= self.max_tokens:
            return [self.max_tokens] * len(clips)

        return [
            max(self.min_tokens * len(clip), raw * self.max_tokens / total_raw_tokens)
            for clip, raw in zip(clips, clip_raw_tokens)
        ]

    def _preprocess(
        self,
        images: Union[ImageInput, VideoInput],
        target_size: List[int],
        merge_size: int = 1,
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Preprocess an image or batch of images. Copy of the `preprocess` method from `CLIPImageProcessor`.

        Args:
            images (`ImageInput`):
                Image or batch of images to preprocess. Expects pixel values ranging from 0 to 255. If pixel values range from 0 to 1, set `do_rescale=False`.
            target_size (`List[int]`):
                The target size to resize the image to. Should be a list of two integers: [target_height, target_width].
            merge_size (`int`, *optional*, defaults to `1`):
                The merge size after the vision encoder.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`PILImageResampling`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the `PILImageResampling` enums.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Scale factor to use if rescaling the image.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Mean to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Standard deviation to use if normalizing the image. Can be a float or a list of floats corresponding to the number of channels in the image.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            data_format (`ChannelDimension`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.   - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.
        """
        images = make_list_of_images(images)

        if do_convert_rgb:
            images = [convert_to_rgb(image) for image in images]

        # All transformations expect numpy arrays.
        images = [to_numpy_array(image) for image in images]

        if is_scaled_image(images[0]) and do_rescale:
            logger.warning_once(
                "It looks like you are trying to rescale already rescaled images. If the input"
                " images have pixel values between 0 and 1, set `do_rescale=False` to avoid rescaling them again."
            )
        if input_data_format is None:
            # We assume that all images have the same channel dimension format.
            input_data_format = infer_channel_dimension_format(images[0])

        height, width = get_image_size(images[0], channel_dim=input_data_format)
        resized_height, resized_width = height, width
        processed_images = []
        for image in images:
            if do_resize:
                resized_height, resized_width = target_size
                image = resize(
                    image, size=(resized_height, resized_width), resample=resample, input_data_format=input_data_format
                )

            if do_rescale:
                image = self.rescale(image, scale=rescale_factor, input_data_format=input_data_format)

            if do_normalize:
                image = self.normalize(
                    image=image, mean=image_mean, std=image_std, input_data_format=input_data_format
                )

            image = to_channel_dimension_format(image, data_format, input_channel_dim=input_data_format)
            processed_images.append(image)

        patches = np.array(processed_images)
        if data_format == ChannelDimension.LAST:
            patches = patches.transpose(0, 3, 1, 2)
        t = patches.shape[0]
        channel = patches.shape[1]
        grid_h, grid_w = resized_height // self.patch_size, resized_width // self.patch_size
        patches = patches.reshape(
            t,
            channel,
            grid_h // merge_size,
            merge_size,
            self.patch_size,
            grid_w // merge_size,
            merge_size,
            self.patch_size,
        )
        patches = patches.transpose(0, 2, 5, 3, 6, 1, 4, 7)
        flatten_patches = patches.reshape(
            t * grid_h * grid_w, channel * self.patch_size * self.patch_size
        )

        return flatten_patches, (t, grid_h, grid_w)

    def preprocess(
        self,
        images: ImageInput,
        do_resize: bool = None,
        resample: PILImageResampling = None,
        do_rescale: bool = None,
        rescale_factor: float = None,
        do_normalize: bool = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None,
        do_convert_rgb: bool = None,
        merge_size: Optional[Union[int, List[int]]] = None,
        frame_types: Optional[Union[int, List[int]]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        data_format: Optional[ChannelDimension] = ChannelDimension.FIRST,
        input_data_format: Optional[Union[str, ChannelDimension]] = None,
    ):
        """
        Args:
            images (`ImageInput`):
                Image to preprocess. Expects a single or batch of images with pixel values ranging from 0 to 255. If
                passing in images with pixel values between 0 and 1, set `do_rescale=False`.
            do_resize (`bool`, *optional*, defaults to `self.do_resize`):
                Whether to resize the image.
            resample (`int`, *optional*, defaults to `self.resample`):
                Resampling filter to use if resizing the image. This can be one of the enum `PILImageResampling`. Only
                has an effect if `do_resize` is set to `True`.
            do_rescale (`bool`, *optional*, defaults to `self.do_rescale`):
                Whether to rescale the image.
            rescale_factor (`float`, *optional*, defaults to `self.rescale_factor`):
                Rescale factor to rescale the image by if `do_rescale` is set to `True`.
            do_normalize (`bool`, *optional*, defaults to `self.do_normalize`):
                Whether to normalize the image.
            image_mean (`float` or `List[float]`, *optional*, defaults to `self.image_mean`):
                Image mean to use for normalization. Only has an effect if `do_normalize` is set to `True`.
            image_std (`float` or `List[float]`, *optional*, defaults to `self.image_std`):
                Image standard deviation to use for normalization. Only has an effect if `do_normalize` is set to
                `True`.
            do_convert_rgb (`bool`, *optional*, defaults to `self.do_convert_rgb`):
                Whether to convert the image to RGB.
            return_tensors (`str` or `TensorType`, *optional*):
                The type of tensors to return. Can be one of:
                - Unset: Return a list of `np.ndarray`.
                - `TensorType.TENSORFLOW` or `'tf'`: Return a batch of type `tf.Tensor`.
                - `TensorType.PYTORCH` or `'pt'`: Return a batch of type `torch.Tensor`.
                - `TensorType.NUMPY` or `'np'`: Return a batch of type `np.ndarray`.
                - `TensorType.JAX` or `'jax'`: Return a batch of type `jax.numpy.ndarray`.
            data_format (`ChannelDimension` or `str`, *optional*, defaults to `ChannelDimension.FIRST`):
                The channel dimension format for the output image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - Unset: Use the channel dimension format of the input image.
            input_data_format (`ChannelDimension` or `str`, *optional*):
                The channel dimension format for the input image. If unset, the channel dimension format is inferred
                from the input image. Can be one of:
                - `"channels_first"` or `ChannelDimension.FIRST`: image in (num_channels, height, width) format.
                - `"channels_last"` or `ChannelDimension.LAST`: image in (height, width, num_channels) format.
                - `"none"` or `ChannelDimension.NONE`: image in (height, width) format.

        """
        do_resize = do_resize if do_resize is not None else self.do_resize
        resample = resample if resample is not None else self.resample
        do_rescale = do_rescale if do_rescale is not None else self.do_rescale
        rescale_factor = rescale_factor if rescale_factor is not None else self.rescale_factor
        do_normalize = do_normalize if do_normalize is not None else self.do_normalize
        image_mean = image_mean if image_mean is not None else self.image_mean
        image_std = image_std if image_std is not None else self.image_std
        merge_size = merge_size if merge_size is not None else self.merge_size
        do_convert_rgb = do_convert_rgb if do_convert_rgb is not None else self.do_convert_rgb

        clips = make_batched_images(images)
        num_clips = len(clips)

        if isinstance(merge_size, (list, tuple)):
            assert len(merge_size) == num_clips, (
                f"merge_size length ({len(merge_size)}) must match number of clips ({num_clips})"
            )
            clip_merge_sizes = list(merge_size)
        else:
            clip_merge_sizes = [merge_size] * num_clips

        if frame_types is None:
            clip_frame_types = [None] * num_clips
        elif isinstance(frame_types, (list, tuple)) and len(frame_types) > 0:
            if isinstance(frame_types[0], (list, tuple)) or frame_types[0] is None:
                assert len(frame_types) == num_clips, (
                    f"frame_types length ({len(frame_types)}) must match number of clips ({num_clips})"
                )
                clip_frame_types = list(frame_types)
            else:
                assert num_clips == 1, "Flat frame_types is only supported for a single clip"
                clip_frame_types = [frame_types]
        else:
            clip_frame_types = [None] * num_clips

        pixel_values, grid_sizes, per_frame_merge_sizes = [], [], []

        clip_max_tokens_list = self._allocate_token_budget(
            clips, clip_merge_sizes, input_data_format,
        )

        for clip, ms, ft, clip_max_tokens in zip(clips, clip_merge_sizes, clip_frame_types, clip_max_tokens_list):
            target_sizes = simple_batched_resize(
                clip,
                factor=self.patch_size * ms,
                min_tokens=self.min_tokens,
                max_tokens=clip_max_tokens,
                input_data_format=input_data_format,
                frame_types=ft,
            )

            for frame, target_size in zip(clip, target_sizes):
                patches, grid_size = self._preprocess(
                    frame,
                    target_size=target_size,
                    merge_size=ms,
                    do_resize=do_resize,
                    resample=resample,
                    do_rescale=do_rescale,
                    rescale_factor=rescale_factor,
                    do_normalize=do_normalize,
                    image_mean=image_mean,
                    image_std=image_std,
                    data_format=data_format,
                    do_convert_rgb=do_convert_rgb,
                    input_data_format=input_data_format,
                )
                pixel_values.append(patches)
                grid_sizes.append(grid_size)
                per_frame_merge_sizes.append(ms)

        pixel_values = np.concatenate(pixel_values, axis=0)
        grid_sizes = np.array(grid_sizes)
        merge_sizes = np.array(per_frame_merge_sizes)

        data = {
            "pixel_values": pixel_values,
            "grid_sizes": grid_sizes,
            "merge_sizes": merge_sizes,
        }

        return BatchFeature(data=data, tensor_type=return_tensors)
