from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageOps


def letterbox_image(
    image: Image.Image,
    size: int = 336,
    fill: tuple[int, int, int] = (128, 128, 128),
) -> Image.Image:
    """EXIF-correct, convert to RGB, preserve aspect ratio, and center pad."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    width, height = image.size
    if width < 1 or height < 1:
        raise ValueError(f"Invalid image size: {image.size}")
    scale = min(size / width, size / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    image = image.resize((new_width, new_height), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
    return canvas


class PenguinLetterboxProcessor:
    """Letterbox first, then delegate rescale/normalize/patchify to Penguin."""

    def __init__(self, processor, image_size: int = 336):
        self.processor = processor
        self.image_size = image_size

    @property
    def neutral_fill(self) -> tuple[int, int, int]:
        mean = getattr(self.processor, "image_mean", (0.5, 0.5, 0.5))
        if isinstance(mean, (float, int)):
            mean = (mean, mean, mean)
        return tuple(max(0, min(255, round(float(value) * 255))) for value in mean)

    def __call__(self, images: Image.Image | Sequence[Image.Image], return_tensors: str = "pt"):
        if isinstance(images, Image.Image):
            images = [images]
        boxed = [letterbox_image(image, self.image_size, self.neutral_fill) for image in images]
        result = self.processor(
            images=boxed,
            do_resize=False,
            merge_size=1,
            return_tensors=return_tensors,
        )
        required = {"pixel_values", "grid_sizes", "merge_sizes"}
        missing = required.difference(result.keys())
        if missing:
            raise RuntimeError(f"Penguin processor missing fields: {sorted(missing)}")
        return result


class PenguinDynamicResolutionProcessor:
    """Use Penguin's native aspect-ratio preserving resize without image padding.

    Penguin already packs differently sized images into one flat patch tensor.
    This wrapper fixes the supported token range, applies EXIF orientation, and
    validates that every emitted token corresponds to a real resized image patch.
    """

    def __init__(
        self,
        processor,
        min_visual_tokens: int = 256,
        max_visual_tokens: int = 1024,
    ):
        if min_visual_tokens < 1 or max_visual_tokens < min_visual_tokens:
            raise ValueError("Invalid dynamic visual-token range")
        self.processor = processor
        self.min_visual_tokens = int(min_visual_tokens)
        self.max_visual_tokens = int(max_visual_tokens)
        # The remote Penguin processor reads these attributes while selecting
        # aligned target sizes. merge_size=1 keeps one encoder token per patch.
        self.processor.min_tokens = self.min_visual_tokens
        self.processor.max_tokens = self.max_visual_tokens

    def _fit_single(
        self,
        image: Image.Image,
        return_tensors: str,
        min_visual_tokens: int | None = None,
        max_visual_tokens: int | None = None,
    ):
        """Fit one real image to the configured patch-token contract.

        Penguin's helper internally multiplies ``min_tokens`` by 1.5 and
        ``max_tokens`` by 0.95 before choosing a grid.  Compensate for those
        implementation factors, then tighten and retry if aspect-ratio rounding
        crosses the public limits.  No canvas or synthetic patch is introduced.
        """

        minimum = int(min_visual_tokens or self.min_visual_tokens)
        maximum = int(max_visual_tokens or self.max_visual_tokens)
        if minimum < 1 or maximum < minimum or maximum > self.max_visual_tokens:
            raise ValueError(f"Invalid per-image token range [{minimum}, {maximum}]")
        original_min = self.processor.min_tokens
        original_max = self.processor.max_tokens
        # Floor is intentional: using ceil(256 / 1.5) makes Penguin's own
        # ``ceil`` alignment jump a square 16x16 grid to 17x17.
        delegate_min = max(1, math.floor(minimum / 1.5))
        delegate_max = max(delegate_min, math.ceil(maximum / 0.95))
        last_count = None
        try:
            for _ in range(12):
                self.processor.min_tokens = delegate_min
                self.processor.max_tokens = delegate_max
                result = self.processor(
                    images=[image],
                    do_resize=True,
                    merge_size=1,
                    return_tensors=return_tensors,
                )
                grids = result["grid_sizes"]
                count = int(grids.prod(dim=1)[0])
                last_count = count
                if minimum <= count <= maximum:
                    return result
                if count > maximum:
                    delegate_max = max(
                        delegate_min,
                        math.floor(
                            delegate_max
                            * maximum
                            / max(count, 1)
                            * 0.98
                        ),
                    )
                else:
                    delegate_min = min(
                        delegate_max,
                        math.ceil(
                            delegate_min
                            * minimum
                            / max(count, 1)
                            * 1.02
                        ),
                    )
            raise RuntimeError(
                "Penguin token-budget tightening did not converge: "
                f"last_count={last_count}, range="
                f"[{minimum}, {maximum}]"
            )
        finally:
            self.processor.min_tokens = original_min
            self.processor.max_tokens = original_max

    def __call__(
        self,
        images: Image.Image | Sequence[Image.Image],
        return_tensors: str = "pt",
        max_visual_tokens: int | Sequence[int] | None = None,
    ):
        if isinstance(images, Image.Image):
            images = [images]
        images = [ImageOps.exif_transpose(image).convert("RGB") for image in images]
        if not images:
            raise ValueError("At least one image is required")
        if any(image.width < 1 or image.height < 1 for image in images):
            raise ValueError("Images must have positive dimensions")
        # Penguin's video-oriented processor shares max_tokens across every
        # item passed in one call. LoopVL defines the budget per image, so
        # select each grid independently and concatenate the already-packed
        # patch tensors afterwards.
        if max_visual_tokens is None:
            caps = [self.max_visual_tokens] * len(images)
        elif isinstance(max_visual_tokens, int):
            caps = [int(max_visual_tokens)] * len(images)
        else:
            caps = [int(value) for value in max_visual_tokens]
            if len(caps) != len(images):
                raise ValueError("One visual-token cap is required per image")
        per_image = [
            self._fit_single(
                image,
                return_tensors,
                min_visual_tokens=self.min_visual_tokens,
                max_visual_tokens=cap,
            )
            for image, cap in zip(images, caps, strict=True)
        ]
        first = per_image[0]
        pixel_values = first["pixel_values"]
        grid_sizes = first["grid_sizes"]
        merge_sizes = first["merge_sizes"]
        if len(per_image) > 1:
            import torch

            pixel_values = torch.cat([item["pixel_values"] for item in per_image], dim=0)
            grid_sizes = torch.cat([item["grid_sizes"] for item in per_image], dim=0)
            merge_sizes = torch.cat([item["merge_sizes"] for item in per_image], dim=0)
        result = {
            "pixel_values": pixel_values,
            "grid_sizes": grid_sizes,
            "merge_sizes": merge_sizes,
        }
        required = {"pixel_values", "grid_sizes", "merge_sizes"}
        missing = required.difference(result.keys())
        if missing:
            raise RuntimeError(f"Penguin processor missing fields: {sorted(missing)}")

        grids = result["grid_sizes"]
        merges = result["merge_sizes"]
        if grids.ndim != 2 or grids.shape[1] != 3 or grids.shape[0] != len(images):
            raise RuntimeError(f"Unexpected Penguin grid_sizes shape: {tuple(grids.shape)}")
        if merges.ndim != 1 or merges.shape[0] != len(images) or bool((merges != 1).any()):
            raise RuntimeError("Dynamic LoopVL input requires merge_size=1 for every image")
        if bool((grids[:, 0] != 1).any()):
            raise RuntimeError("LoopVL supports one image frame per sample")

        counts = grids.prod(dim=1)
        import torch

        cap_tensor = torch.tensor(caps, device=counts.device, dtype=counts.dtype)
        if bool(((counts < self.min_visual_tokens) | (counts > cap_tensor)).any()):
            raise RuntimeError(
                "Penguin emitted visual token counts outside per-image budgets: "
                f"counts={counts.tolist()}, caps={caps}"
            )
        pixel_values = result["pixel_values"]
        if pixel_values.ndim != 2 or pixel_values.shape[0] != int(counts.sum()):
            raise RuntimeError(
                "Packed pixel patches do not exactly match the real dynamic grids: "
                f"patches={tuple(pixel_values.shape)}, grids={grids.tolist()}"
            )
        return result


def build_penguin_image_processor(processor, vision_config):
    if getattr(vision_config, "dynamic_resolution", False):
        return PenguinDynamicResolutionProcessor(
            processor,
            min_visual_tokens=vision_config.min_visual_tokens,
            max_visual_tokens=vision_config.max_visual_tokens,
        )
    return PenguinLetterboxProcessor(processor, vision_config.input_size)
