from __future__ import annotations

import numpy as np
import torch
from shapely import STRtree, box

from sahi.logger import logger
from sahi.postprocess.utils import ObjectPredictionList, has_match, merge_object_prediction_pair
from sahi.prediction import ObjectPrediction
from sahi.utils.import_utils import check_requirements


def batched_nms(predictions: torch.tensor, match_metric: str = "IOU", match_threshold: float = 0.5):
    """Apply non-maximum suppression to avoid detecting too many overlapping bounding boxes for a given object.

    Args:
        predictions: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        A list of filtered indexes, Shape: [ ,]
    """

    scores = predictions[:, 4].squeeze()
    category_ids = predictions[:, 5].squeeze()
    keep_mask = torch.zeros_like(category_ids, dtype=torch.bool)
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_indices = nms(predictions[curr_indices], match_metric, match_threshold)
        keep_mask[curr_indices[curr_keep_indices]] = True
    keep_indices = torch.where(keep_mask)[0]
    # sort selected indices by their scores
    keep_indices = keep_indices[scores[keep_indices].sort(descending=True)[1]].tolist()
    return keep_indices


def nms(
    predictions: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Optimized non-maximum suppression for axis-aligned bounding boxes using STRTree.

    Args:
        predictions: (tensor) The location preds for the image along with the class
            predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.

    Returns:
        A list of filtered indexes, Shape: [ ,]
    """
    if len(predictions) == 0:
        return []

    # Ensure predictions are on CPU and convert to numpy
    if predictions.device.type != "cpu":
        predictions = predictions.cpu()

    predictions_np = predictions.numpy()

    # Extract coordinates and scores
    x1 = predictions_np[:, 0]
    y1 = predictions_np[:, 1]
    x2 = predictions_np[:, 2]
    y2 = predictions_np[:, 3]
    scores = predictions_np[:, 4]

    # Calculate areas
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes (vectorized)
    boxes = box(x1, y1, x2, y2)

    # Sort indices by score (descending)
    sorted_idxs = np.argsort(scores)[::-1]

    # Build STRtree
    tree = STRtree(boxes)

    keep = []
    suppressed = set()

    for current_idx in sorted_idxs:
        if current_idx in suppressed:
            continue

        keep.append(current_idx)
        current_box = boxes[current_idx]
        current_area = areas[current_idx]

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx or candidate_idx in suppressed:
                continue

            # Skip candidates with higher scores (already processed)
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                # Use box coordinates for stable ordering
                current_coords = (
                    x1[current_idx],
                    y1[current_idx],
                    x2[current_idx],
                    y2[current_idx],
                )
                candidate_coords = (
                    x1[candidate_idx],
                    y1[candidate_idx],
                    x2[candidate_idx],
                    y2[candidate_idx],
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx] - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx])
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError("Invalid match_metric")

            # Suppress if overlap exceeds threshold
            if metric >= match_threshold:
                suppressed.add(candidate_idx)

    return keep


def batched_greedy_nmm(
    object_predictions_as_tensor: torch.tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """Apply greedy version of non-maximum merging per category to avoid detecting too many overlapping bounding boxes
    for a given object.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = {}
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = greedy_nmm(object_predictions_as_tensor[curr_indices], match_metric, match_threshold)
        curr_indices_list = curr_indices.tolist()
        keep_to_merge_list.update(_remap_indices(curr_keep_to_merge_list, curr_indices_list))
    return keep_to_merge_list


def greedy_nmm(
    object_predictions_as_tensor: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """
    Optimized greedy non-maximum merging for axis-aligned bounding boxes using STRTree.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.
    Returns:
        keep_to_merge_list: (dict[int, list[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    # Extract coordinates and scores as tensors
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]
    scores = object_predictions_as_tensor[:, 4]

    # Calculate areas as tensor (vectorized operation)
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes only once
    boxes = []
    for i in range(len(object_predictions_as_tensor)):
        boxes.append(
            box(
                x1[i].item(),  # Convert only individual values
                y1[i].item(),
                x2[i].item(),
                y2[i].item(),
            )
        )

    # Sort indices by score (descending) using torch
    sorted_idxs = torch.argsort(scores, descending=True).tolist()

    # Build STRtree
    tree = STRtree(boxes)

    keep_to_merge_list = {}
    suppressed = set()

    for current_idx in sorted_idxs:
        if current_idx in suppressed:
            continue

        current_box = boxes[current_idx]
        current_area = areas[current_idx].item()  # Convert only when needed

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        merge_list = []
        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx or candidate_idx in suppressed:
                continue

            # Only consider candidates with lower or equal score
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                # Use box coordinates for stable ordering
                current_coords = (
                    x1[current_idx].item(),
                    y1[current_idx].item(),
                    x2[current_idx].item(),
                    y2[current_idx].item(),
                )
                candidate_coords = (
                    x1[candidate_idx].item(),
                    y1[candidate_idx].item(),
                    x2[candidate_idx].item(),
                    y2[candidate_idx].item(),
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx].item() - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx].item())
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError("Invalid match_metric")

            # Add to merge list if overlap exceeds threshold
            if metric >= match_threshold:
                merge_list.append(candidate_idx)
                suppressed.add(candidate_idx)

        keep_to_merge_list[int(current_idx)] = [int(idx) for idx in merge_list]

    return keep_to_merge_list


def _update_merge_mappings(
    current_idx: int,
    matched_box_indices: list[int],
    keep_to_merge_list: dict[int, list[int]],
    merge_to_keep: dict[int, int],
):
    """Helper function to update merge mapping dictionaries.

    Args:
        current_idx: Index of the current box being processed
        matched_box_indices: List of indices that match with current box
        keep_to_merge_list: Dictionary mapping keep indices to merge lists
        merge_to_keep: Dictionary mapping merge indices to their keep index
    """
    if current_idx not in merge_to_keep:
        keep_to_merge_list[current_idx] = []

        for matched_box_idx in matched_box_indices:
            matched_box_idx_native = int(matched_box_idx)
            if matched_box_idx_native not in merge_to_keep:
                keep_to_merge_list[current_idx].append(matched_box_idx_native)
                merge_to_keep[matched_box_idx_native] = current_idx
    else:
        keep_idx = merge_to_keep[current_idx]
        for matched_box_idx in matched_box_indices:
            matched_box_idx_native = int(matched_box_idx)
            if (
                matched_box_idx_native not in keep_to_merge_list.get(keep_idx, [])
                and matched_box_idx_native not in merge_to_keep
            ):
                if keep_idx not in keep_to_merge_list:
                    keep_to_merge_list[keep_idx] = []
                keep_to_merge_list[keep_idx].append(matched_box_idx_native)
                merge_to_keep[matched_box_idx_native] = keep_idx


def _remap_indices(
    local_keep_to_merge_list: dict[int, list[int]],
    global_indices: list[int],
) -> dict[int, list[int]]:
    """Helper function to remap local indices to global indices.

    Args:
        local_keep_to_merge_list: Dictionary with local indices (within category)
        global_indices: List mapping local indices to global indices

    Returns:
        Dictionary with remapped global indices
    """
    global_keep_to_merge_list = {}
    for local_keep, local_merge_list in local_keep_to_merge_list.items():
        global_keep = global_indices[local_keep]
        global_merge_list = [global_indices[local_merge_idx] for local_merge_idx in local_merge_list]
        global_keep_to_merge_list[global_keep] = global_merge_list
    return global_keep_to_merge_list


def batched_nmm(
    object_predictions_as_tensor: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """Apply non-maximum merging per category to avoid detecting too many overlapping bounding boxes for a given object.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for
            match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    category_ids = object_predictions_as_tensor[:, 5].squeeze()
    keep_to_merge_list = {}
    for category_id in torch.unique(category_ids):
        curr_indices = torch.where(category_ids == category_id)[0]
        curr_keep_to_merge_list = nmm(object_predictions_as_tensor[curr_indices], match_metric, match_threshold)
        curr_indices_list = curr_indices.tolist()
        keep_to_merge_list.update(_remap_indices(curr_keep_to_merge_list, curr_indices_list))
    return keep_to_merge_list


def nmm(
    object_predictions_as_tensor: torch.Tensor,
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
):
    """Apply non-maximum merging to avoid detecting too many overlapping bounding boxes for a given object.

    Args:
        object_predictions_as_tensor: (tensor) The location preds for the image
            along with the class predscores, Shape: [num_boxes,5].
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    # Extract coordinates and scores as tensors
    x1 = object_predictions_as_tensor[:, 0]
    y1 = object_predictions_as_tensor[:, 1]
    x2 = object_predictions_as_tensor[:, 2]
    y2 = object_predictions_as_tensor[:, 3]
    scores = object_predictions_as_tensor[:, 4]

    # Calculate areas as tensor (vectorized operation)
    areas = (x2 - x1) * (y2 - y1)

    # Create Shapely boxes only once
    boxes = []
    for i in range(len(object_predictions_as_tensor)):
        boxes.append(
            box(
                x1[i].item(),  # Convert only individual values
                y1[i].item(),
                x2[i].item(),
                y2[i].item(),
            )
        )

    # Sort indices by score (descending) using torch
    sorted_idxs = torch.argsort(scores, descending=True).tolist()

    # Build STRtree
    tree = STRtree(boxes)

    keep_to_merge_list = {}
    merge_to_keep = {}

    for current_idx in sorted_idxs:
        current_box = boxes[current_idx]
        current_area = areas[current_idx].item()  # Convert only when needed

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_box)

        matched_box_indices = []
        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx:
                continue

            # Only consider candidates with lower or equal score
            if scores[candidate_idx] > scores[current_idx]:
                continue

            # For equal scores, use deterministic tie-breaking based on box coordinates
            if scores[candidate_idx] == scores[current_idx]:
                # Use box coordinates for stable ordering
                current_coords = (
                    x1[current_idx].item(),
                    y1[current_idx].item(),
                    x2[current_idx].item(),
                    y2[current_idx].item(),
                )
                candidate_coords = (
                    x1[candidate_idx].item(),
                    y1[candidate_idx].item(),
                    x2[candidate_idx].item(),
                    y2[candidate_idx].item(),
                )

                # Compare coordinates lexicographically
                if candidate_coords > current_coords:
                    continue

            # Calculate intersection area
            candidate_box = boxes[candidate_idx]
            intersection = current_box.intersection(candidate_box).area

            # Calculate metric
            if match_metric == "IOU":
                union = current_area + areas[candidate_idx].item() - intersection
                metric = intersection / union if union > 0 else 0
            elif match_metric == "IOS":
                smaller = min(current_area, areas[candidate_idx].item())
                metric = intersection / smaller if smaller > 0 else 0
            else:
                raise ValueError("Invalid match_metric")

            # Add to matched list if overlap exceeds threshold
            if metric >= match_threshold:
                matched_box_indices.append(candidate_idx)

        # Update merge mappings using helper function
        current_idx_native = int(current_idx)
        _update_merge_mappings(current_idx_native, matched_box_indices, keep_to_merge_list, merge_to_keep)

    return keep_to_merge_list


def _mask_match(
    multipolygon1,
    multipolygon2,
    match_metric: str,
    match_threshold: float,
    use_largest_polygon_only: bool = False,
) -> bool:
    """Check if two mask geometries match using polygon overlap.

    Args:
        multipolygon1: First shapely multipolygon geometry
        multipolygon2: Second shapely multipolygon geometry
        match_metric: (str) IOU, IOS, or IAREA (intersection area)
        match_threshold: (float) The overlap threshold for match metric
        use_largest_polygon_only: (bool) If True, only use the largest polygon from each multipolygon for matching

    Returns:
        bool: True if geometries match according to metric and threshold
    """
    # Extract largest polygons if requested
    if use_largest_polygon_only:
        poly1 = max(multipolygon1.geoms, key=lambda p: p.area) if multipolygon1.geoms else multipolygon1
        poly2 = max(multipolygon2.geoms, key=lambda p: p.area) if multipolygon2.geoms else multipolygon2
    else:
        poly1 = multipolygon1
        poly2 = multipolygon2

    intersection = poly1.intersection(poly2).area

    if match_metric == "IOU":
        area1 = poly1.area
        area2 = poly2.area
        union = area1 + area2 - intersection
        metric = intersection / union if union > 0 else 0
    elif match_metric == "IOS":
        area1 = poly1.area
        area2 = poly2.area
        smaller = min(area1, area2)
        metric = intersection / smaller if smaller > 0 else 0
    elif match_metric == "IAREA":
        metric = intersection
    else:
        raise ValueError(f"Invalid match_metric: {match_metric}")

    return metric >= match_threshold


def mask_nmm(
    object_predictions: list[ObjectPrediction],
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
    use_largest_polygon_only: bool = False,
):
    """Apply non-maximum merging to mask predictions using polygon geometries.

    Args:
        object_predictions: List of ObjectPrediction instances with mask/polygon data.
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.
        use_largest_polygon_only: (bool) If True, only use the largest polygon from each multipolygon for matching.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    scores = [obj_pred.score.value for obj_pred in object_predictions]
    shapely_annotations = [obj_pred.to_shapely_annotation() for obj_pred in object_predictions]
    multipolygons = [ann.multipolygon for ann in shapely_annotations]

    # Sort indices by score (descending)
    sorted_idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    # Build STRtree for efficient spatial queries
    tree = STRtree(multipolygons)

    keep_to_merge_list = {}
    merge_to_keep = {}

    for current_idx in sorted_idxs:
        current_multipolygon = multipolygons[current_idx]
        current_score = scores[current_idx]

        # Query potential intersections using STRtree
        candidate_idxs = tree.query(current_multipolygon)

        matched_box_indices = []
        for candidate_idx in candidate_idxs:
            if candidate_idx == current_idx:
                continue

            # Only consider candidates with lower or equal score
            if scores[candidate_idx] > current_score:
                continue

            # For equal scores, use deterministic tie-breaking based on index
            if scores[candidate_idx] == current_score:
                if candidate_idx > current_idx:
                    continue

            # Check if masks match using helper function
            if _mask_match(
                current_multipolygon,
                multipolygons[candidate_idx],
                match_metric,
                match_threshold,
                use_largest_polygon_only,
            ):
                matched_box_indices.append(candidate_idx)

        # Update merge mappings using helper function
        _update_merge_mappings(current_idx, matched_box_indices, keep_to_merge_list, merge_to_keep)

    return keep_to_merge_list


def batched_mask_nmm(
    object_predictions: list[ObjectPrediction],
    match_metric: str = "IOU",
    match_threshold: float = 0.5,
    use_largest_polygon_only: bool = False,
):
    """Apply non-maximum merging per category for mask predictions.

    Args:
        object_predictions: List of ObjectPrediction instances with mask/polygon data.
        match_metric: (str) IOU or IOS
        match_threshold: (float) The overlap thresh for match metric.
        use_largest_polygon_only: (bool) If True, only use the largest polygon from each multipolygon for matching.
    Returns:
        keep_to_merge_list: (Dict[int:List[int]]) mapping from prediction indices
        to keep to a list of prediction indices to be merged.
    """
    category_ids = [pred.category.id for pred in object_predictions]
    keep_to_merge_list = {}

    for category_id in set(category_ids):
        curr_indices = [i for i, cat_id in enumerate(category_ids) if cat_id == category_id]
        curr_predictions = [object_predictions[i] for i in curr_indices]
        curr_keep_to_merge_list = mask_nmm(curr_predictions, match_metric, match_threshold, use_largest_polygon_only)
        keep_to_merge_list.update(_remap_indices(curr_keep_to_merge_list, curr_indices))

    return keep_to_merge_list


class PostprocessPredictions:
    """Utilities for calculating IOU/IOS based match for given ObjectPredictions."""

    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
    ):
        self.match_threshold = match_threshold
        self.class_agnostic = class_agnostic
        self.match_metric = match_metric

        check_requirements(["torch"])

    def __call__(self, predictions: list[ObjectPrediction]):
        raise NotImplementedError()


class NMSPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep = nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, match_metric=self.match_metric
            )
        else:
            keep = batched_nms(
                object_predictions_as_torch, match_threshold=self.match_threshold, match_metric=self.match_metric
            )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions


class NMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep_to_merge_list = nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep_to_merge_list = batched_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    self.match_metric,
                    self.match_threshold,
                ):
                    object_prediction_list[keep_ind] = merge_object_prediction_pair(
                        object_prediction_list[keep_ind].tolist(), object_prediction_list[merge_ind].tolist()
                    )
            selected_object_predictions.append(object_prediction_list[keep_ind].tolist())

        return selected_object_predictions


class GreedyNMMPostprocess(PostprocessPredictions):
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_torch = object_prediction_list.totensor()
        if self.class_agnostic:
            keep_to_merge_list = greedy_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )
        else:
            keep_to_merge_list = batched_greedy_nmm(
                object_predictions_as_torch,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
            )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                if has_match(
                    object_prediction_list[keep_ind].tolist(),
                    object_prediction_list[merge_ind].tolist(),
                    self.match_metric,
                    self.match_threshold,
                ):
                    object_prediction_list[keep_ind] = merge_object_prediction_pair(
                        object_prediction_list[keep_ind].tolist(), object_prediction_list[merge_ind].tolist()
                    )
            selected_object_predictions.append(object_prediction_list[keep_ind].tolist())

        return selected_object_predictions


class LSNMSPostprocess(PostprocessPredictions):
    # https://github.com/remydubois/lsnms/blob/10b8165893db5bfea4a7cb23e268a502b35883cf/lsnms/nms.py#L62
    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        try:
            from lsnms import nms
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                'Please run "pip install lsnms>0.3.1" to install lsnms first for lsnms utilities.'
            )

        if self.match_metric == "IOS":
            NotImplementedError(f"match_metric={self.match_metric} is not supported for LSNMSPostprocess")

        logger.warning("LSNMSPostprocess is experimental and not recommended to use.")

        object_prediction_list = ObjectPredictionList(object_predictions)
        object_predictions_as_numpy = object_prediction_list.tonumpy()

        boxes = object_predictions_as_numpy[:, :4]
        scores = object_predictions_as_numpy[:, 4]
        class_ids = object_predictions_as_numpy[:, 5].astype("uint8")

        keep = nms(
            boxes, scores, iou_threshold=self.match_threshold, class_ids=None if self.class_agnostic else class_ids
        )

        selected_object_predictions = object_prediction_list[keep].tolist()
        if not isinstance(selected_object_predictions, list):
            selected_object_predictions = [selected_object_predictions]

        return selected_object_predictions


class MaskNMMPostprocess(PostprocessPredictions):
    def __init__(
        self,
        match_threshold: float = 0.5,
        match_metric: str = "IOU",
        class_agnostic: bool = True,
        use_largest_polygon_only: bool = False,
    ):
        super().__init__(match_threshold, match_metric, class_agnostic)
        self.use_largest_polygon_only = use_largest_polygon_only

    def __call__(
        self,
        object_predictions: list[ObjectPrediction],
    ):
        if self.class_agnostic:
            keep_to_merge_list = mask_nmm(
                object_predictions,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
                use_largest_polygon_only=self.use_largest_polygon_only,
            )
        else:
            keep_to_merge_list = batched_mask_nmm(
                object_predictions,
                match_threshold=self.match_threshold,
                match_metric=self.match_metric,
                use_largest_polygon_only=self.use_largest_polygon_only,
            )

        selected_object_predictions = []
        for keep_ind, merge_ind_list in keep_to_merge_list.items():
            for merge_ind in merge_ind_list:
                # defensive check
                ann_keep = object_predictions[keep_ind].to_shapely_annotation()
                ann_merge = object_predictions[merge_ind].to_shapely_annotation()
                if _mask_match(
                    ann_keep.multipolygon,
                    ann_merge.multipolygon,
                    self.match_metric,
                    self.match_threshold,
                    self.use_largest_polygon_only,
                ):
                    object_predictions[keep_ind] = merge_object_prediction_pair(
                        object_predictions[keep_ind], object_predictions[merge_ind]
                    )
            selected_object_predictions.append(object_predictions[keep_ind])

        return selected_object_predictions
