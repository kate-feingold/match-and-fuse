from pathlib import Path
from PIL import Image

import torch

from src.maf_flux.constants import GRID_PREFIX, QUALITY


def vgrid_pil(*images, gap: int = 0):
    if not images:
        raise ValueError("At least one image must be provided.")

    base_width = max([img.width for img in images])
    total_height = sum(img.height for img in images) + gap * (len(images) - 1)
    grid_img = Image.new('RGB', (base_width, total_height), (255, 255, 255))
    y_offset = 0
    for img in images:
        grid_img.paste(img, (0, y_offset))
        y_offset += img.height
        y_offset += gap

    return grid_img


def hgrid_pil(*images, gap: int = 0):
    if not images:
        raise ValueError("At least one image must be provided.")

    base_height = max([img.height for img in images])
    total_width = sum(img.width for img in images) + gap * (len(images) - 1)
    grid_img = Image.new('RGB', (total_width, base_height), (255, 255, 255))
    x_offset = 0
    for img in images:
        grid_img.paste(img, (x_offset, 0))
        x_offset += img.width
        x_offset += gap

    return grid_img


def shift_match_coords(matches_grid, shift_coord: int, shift_value):
    unmached_mask = (matches_grid[..., 0] == -1) | (matches_grid[..., 1] == -1)
    matches_grid[..., shift_coord] += shift_value  # Shift y coordinates for top image matches
    matches_grid[unmached_mask] = -1
    return matches_grid


def convert_matches_to_grid(matches_grid, grid_horizontal: bool = True):
    """
    Convert matches_grid to [kv_h, kv_w, 2] format where matches correspond to positions in the combined grid image.

    Args:
        matches_grid: Tensor of matches with shape:
            - If grid_horizontal: [2, kv_h, kv_w//2, 2]
            - If not grid_horizontal: [2, kv_h//2, kv_w, 2]
        grid_horizontal: Whether the two images are placed side-by-side (True) or top-bottom (False)

    Returns:
        Tensor of shape [kv_h, kv_w, 2] containing matching coordinates in the combined grid
    """
    if grid_horizontal:
        kv_h = matches_grid.shape[1]
        half_kv_w = matches_grid.shape[2]
        kv_w = half_kv_w * 2
        matches_combined = torch.zeros((kv_h, kv_w, 2), dtype=torch.long)

        # Left half - matches from matches_grid[0] with x shifted by half_kv_w
        matches_left = matches_grid[0].clone()
        matches_left = shift_match_coords(matches_left, 0, half_kv_w)
        matches_combined[:, :half_kv_w] = matches_left
        # Right half - matches from matches_grid[1] with no shift
        matches_combined[:, half_kv_w:] = matches_grid[1]
    else:  # vertical grid
        half_kv_h = matches_grid.shape[1]  # Half height
        kv_w = matches_grid.shape[2]  # Full width
        kv_h = half_kv_h * 2  # Full height
        matches_combined = torch.zeros((kv_h, kv_w, 2), dtype=torch.long)

        # Top half - matches from matches_grid[0] with y shifted by half_kv_h
        matches_top = matches_grid[0].clone()
        matches_top = shift_match_coords(matches_top, 1, half_kv_h)
        matches_combined[:half_kv_h] = matches_top
        # Bottom half - matches from matches_grid[1] with no shift
        matches_combined[half_kv_h:] = matches_grid[1]

    return matches_combined


def image_inds_in_grid(width, height, vertical: bool = False):
    """
    Ids of tokens of one of the images in a grid of 2 images.
    Args:
        vertical: whether the grid image was stacked vertically or horizontally
    """
    sequence_ids = torch.arange(0, width * height, dtype=torch.long).view(height, width)
    if vertical:
        half_ids_1st = sequence_ids[:height // 2]
        half_ids_2nd = sequence_ids[height // 2:]
    else:
        half_ids_1st = sequence_ids[:, :width // 2]
        half_ids_2nd = sequence_ids[:, width // 2:]
    return half_ids_1st.flatten(), half_ids_2nd.flatten()


def define_graph(num_images, max_adjacent_nodes: int = None):
    def adjacency_sparse(graph, d):
        return [
            edge for edge in graph
            if min((edge[1] - edge[0]) % num_images, (edge[0] - edge[1]) % num_images) <= d // 2
        ]

    graph_full = [[i, j] for i in range(num_images) for j in range(i + 1, num_images)]

    if max_adjacent_nodes is None:
        graph = graph_full
    else:
        assert max_adjacent_nodes > 0 and max_adjacent_nodes % 2 == 0, f"Adjacency must be a positive even number, got {max_adjacent_nodes}"
        graph = adjacency_sparse(graph_full, max_adjacent_nodes)
    edge_presence = {
        img_id: [(idx, int(edge[1] == img_id)) for idx, edge in enumerate(graph) if img_id in edge]
        for img_id in range(num_images)
    }
    return graph, edge_presence


def get_num_vertices(num_edges, max_adjacent_nodes: int = None):
    if max_adjacent_nodes is None:  # fully connected graph
        return int((1 + (1 + 8 * num_edges) ** 0.5) / 2)
    else:
        return int(num_edges // (max_adjacent_nodes / 2))


def image_path_to_prompt(
    image_names: list[str], img_wh: tuple[int, int], use_pair_graph: bool,
    caption: dict, max_adjacent_nodes: int = None, mode: str = "edit", flowedit: bool = False,
) -> list[str]:
    def _lower(s): return s[0].lower() + s[1:]
    def _upper(s): return s[0].upper() + s[1:]

    base_key = "src" if mode == "src" else "edit"
    per_image_key = "per_image_non_shared_src" if (mode == "src" or flowedit) else "per_image_non_shared_edit"

    base_prompt = caption[base_key]
    per_image_pose_bg = caption.get(per_image_key)

    if per_image_pose_bg is None:
        shared = f"{base_prompt} {QUALITY}" if not use_pair_graph else f"{GRID_PREFIX} {_lower(base_prompt)} {QUALITY}"
        return [shared]

    stems = [Path(img).stem for img in image_names]
    if not use_pair_graph:
        return [f"{base_prompt} {_upper(per_image_pose_bg[s])}. {QUALITY}" for s in stems]
    else:
        pos1, pos2 = ("top", "bottom") if img_wh[0] > img_wh[1] else ("left", "right")
        graph, _ = define_graph(len(image_names), max_adjacent_nodes=max_adjacent_nodes)
        return [
            f"{GRID_PREFIX} {_lower(base_prompt)} In the {pos1} image, {per_image_pose_bg[stems[a]]}. In the {pos2} image, {per_image_pose_bg[stems[b]]}. {QUALITY}"
            for a, b in graph
        ]


def load_matches(match_folder, img_paths, i_from: int, i_to: int, grid_matches: bool=False,
                 grid_horizontal=False, flatten: bool = False):
    img_names = [Path(img).stem for img in img_paths]
    # Load matches of i_from -> i_to or swap(i_to, i_from)
    match_path = match_folder / f"{img_names[i_from]}_{img_names[i_to]}_matches.pth"
    swap_match_path = match_folder / f"{img_names[i_to]}_{img_names[i_from]}_matches.pth"
    if match_path.exists():
        matches_grid = torch.load(match_path).to(torch.long)
    elif swap_match_path.exists():
        matches_grid = torch.load(swap_match_path).to(torch.long).flip(0)
    else:
        raise RuntimeError(f"{match_path} or {swap_match_path} don't exist")

    # Fix unresolved matching bug
    matches_grid[:, 0, 0] = -1

    if grid_matches:
        matches_grid = convert_matches_to_grid(matches_grid, grid_horizontal=grid_horizontal).unsqueeze(0)
    else:
        matches_grid = matches_grid.unsqueeze(0)

    if flatten:
        matches = matches_grid[..., 1] * matches_grid.shape[-2] + matches_grid[..., 0]  # y * q_width + x
        matches = matches.flatten(-2)  # (2, q_seq_length)
    else:
        matches = matches_grid
    return matches
