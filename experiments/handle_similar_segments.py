"""Handle similar segments."""

import os

from PIL import Image

from openadapt import adapters, cache, config, plotting, utils, vision
from openadapt.custom_logger import logger

DEBUG = True
MIN_SEGMENT_SSIM = 0.95  # threshold for considering segments structurally similar
MIN_SEGMENT_SIZE_SIM = 0.95  # threshold for considering segment sizes similar


# TODO: consolidate with strategies.visual.get_window_segmentation
@cache.cache(enabled=not DEBUG)
def get_similar_segment_groups(
    image_file_path: str,
    min_segment_ssim: float = MIN_SEGMENT_SSIM,
    min_segment_size_sim: float = MIN_SEGMENT_SIZE_SIM,
    show_images: bool = DEBUG,
    contrast_factor: int = 10000,
) -> tuple:
    """Get similar segment groups."""
    image = Image.open(image_file_path)
    image.show()

    if contrast_factor:
        image = utils.increase_contrast(image, contrast_factor)
        image.show()

    segmentation_adapter = adapters.get_default_segmentation_adapter()
    segmented_image = segmentation_adapter.fetch_segmented_image(image)
    if show_images:
        segmented_image.show()

    import ipdb

    ipdb.set_trace()

    masks = vision.get_masks_from_segmented_image(segmented_image)
    logger.info(f"{len(masks)=}")
    if show_images:
        plotting.display_binary_images_grid(masks)

    refined_masks = vision.refine_masks(masks)
    logger.info(f"{len(refined_masks)=}")
    if show_images:
        plotting.display_binary_images_grid(refined_masks)

    masked_images = vision.extract_masked_images(image, refined_masks)
    descriptions = ["" for _ in masked_images]
    if show_images:
        plotting.display_images_table_with_titles(masked_images, descriptions)

    similar_idx_groups, ungrouped_idxs, ssim_matrix, _ = vision.get_similar_image_idxs(
        masked_images,
        min_segment_ssim,
        min_segment_size_sim,
    )
    logger.info(f"{len(similar_idx_groups)=}")

    return (
        image,
        masked_images,
        refined_masks,
        similar_idx_groups,
        ungrouped_idxs,
        ssim_matrix,
    )


def main() -> None:
    """Main."""
    image_file_path = os.path.join(config.ROOT_DIR_PATH, "../tests/assets/excel.png")

    MAX_GROUPS = 2

    for min_segment_ssim in (MIN_SEGMENT_SSIM, MIN_SEGMENT_SSIM // 3):
        logger.info(f"{min_segment_ssim=}")
        image, masked_images, masks, similar_idx_groups, ungrouped_idxs, ssim_matrix = (
            get_similar_segment_groups(image_file_path)
        )
        similar_idx_groups = sorted(
            similar_idx_groups,
            key=lambda group: len(group),
            reverse=True,
        )
        if MAX_GROUPS:
            similar_idx_groups = similar_idx_groups[:MAX_GROUPS]
        plotting.plot_similar_image_groups(
            masked_images,
            similar_idx_groups,
            ssim_matrix,
            [
                f"min_ssim={MIN_SEGMENT_SSIM}",
                f"min_size_sim={MIN_SEGMENT_SIZE_SIM}",
            ],
        )

        """
        - images:
            - original
            - one segment mask
            - multiple segment masks
            - original with one segment highlighted
            - original with multiple segments highlighted
            - original with one segment labelled
            - original with multiple segments labelled
            - original with one segment highlighted+labelled
            - original with multiple segments highlighted+labelled
            - individual segment
            - individual segment labelled
        - one or multiple segments per prompt
        """
        for similar_idx_group in similar_idx_groups:
            similar_masks = [masks[idx] for idx in similar_idx_group]
            highlighted_image = plotting.highlight_masks(image, similar_masks)
            highlighted_image.show()

        import ipdb

        ipdb.set_trace()
        foo = 1  # noqa


if __name__ == "__main__":
    main()
