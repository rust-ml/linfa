use crate::appx_dbscan::hyperparameters::AppxDbscanHyperParams;
use linfa::Float;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use ndarray_stats::DeviationExt;
use std::collections::HashMap;

#[derive(PartialEq, Debug)]
pub enum IntersectionType {
    FullyCovered,
    Disjoint,
    Intersecting,
}

#[derive(Clone)]
/// Tree structure that divides the space in nested cells to perform approximate range counting
/// Each member of this structure is a node in the tree
pub struct TreeStructure<F: Float> {
    /// The index of the cell represented by this node
    cell_center: Array1<F>,
    /// The size of the cell
    side_size: F,
    /// The number of points cointained in the cell
    cnt: usize,
    /// The collection of nested sub-cells (bounded by 2^D at max, with D constant)
    children: HashMap<Array1<i64>, TreeStructure<F>>,
}

impl<F: Float> TreeStructure<F> {
    pub fn new(cell_index: &Array1<i64>, side_size: F) -> TreeStructure<F> {
        let structure = TreeStructure {
            cell_center: cell_center_from_cell_index(cell_index.view(), side_size),
            cnt: 0,
            side_size: side_size,
            children: HashMap::new(),
        };
        structure
    }

    pub fn new_empty() -> TreeStructure<F> {
        TreeStructure {
            cell_center: Array1::zeros(1),
            cnt: 0,
            side_size: F::from(0.0).unwrap(),
            children: HashMap::with_capacity(0),
        }
    }

    /// Generates a tree starting from the points given in input. To function correctly, the points in input
    /// must be all and only the core points in a given cell of the approximated DBSCAN algorithm with side size
    /// equal to `tolerance/sqrt(D)`. This is assumed true during the construction.
    pub fn build_structure(
        points: Vec<ArrayView1<F>>,
        params: &AppxDbscanHyperParams<F>,
    ) -> TreeStructure<F> {
        if points.len() == 0 {
            panic!("AppxDbscan::build structure internal error: attempting to initialize counting tree with no points");
        }
        let dimensionality = points[0].dim();
        let base_side_size = params.tolerance() / (F::from(dimensionality).unwrap()).sqrt();
        let levels_count = F::from(1.0).unwrap()
            + (F::from(1.0).unwrap() / params.slack())
                .log(F::from(2.0).unwrap())
                .ceil();
        let levels_count = if levels_count < F::from(1.0).unwrap() {
            1
        } else {
            levels_count.to_i32().unwrap()
        };
        // The approximated DBSCAN algorithm needs one instance of this structure for every core cell.
        // This gives that all the points in input are contained in the cell of side size `epsilon/sqrt(D)`.
        // All the points can then be added to the root and we proceed directly to divide the core cell in its sub-cells
        let mut root = TreeStructure::new(&get_base_cell_index(&points[0], params), base_side_size);
        root.cnt = points.len();

        for point in &points {
            let mut curr_side_size = base_side_size;
            let mut prev_child = &mut root;
            //il livello 0 è occupato dalla radice
            for _ in 1..=levels_count {
                curr_side_size = curr_side_size / F::from(2.0).unwrap();
                let index_arr = get_cell_index(point, curr_side_size);
                let curr_child: &mut TreeStructure<F> = prev_child
                    .children
                    .entry(index_arr.clone())
                    .or_insert(TreeStructure::new(&index_arr, curr_side_size));
                curr_child.cnt += 1;
                prev_child = curr_child;
            }
        }
        root
    }

    /// Performs the approximated range counting on the tree given the point in input. It stops as soon as the counting
    /// is non zero, so the result is not actually the exact count but rather 0 if there is no point in the tree
    /// in the vicinity of `q`, and a value that is less or equal to the number of points in the vicinity of `q` otherwise.
    /// The points in the vicinity are found for certain if they are at a distance less than equal to `epsilon` from `q` and
    /// are excluded for certain if their distance from `q` is greater than `epsilon(1 + rho)`. All the points in between are
    /// counted in an arbitrary way, depending on what is more efficient.
    pub fn approximate_range_counting(
        &self,
        q: &ArrayView1<F>,
        params: &AppxDbscanHyperParams<F>,
    ) -> usize {
        let mut ans: usize = 0;
        let intersection_type =
            determine_intersection(q, params, &self.cell_center, self.side_size);
        match intersection_type {
            IntersectionType::Disjoint => {}
            IntersectionType::FullyCovered => {
                ans += self.cnt;
            }
            IntersectionType::Intersecting => {
                if self.children.len() > 0 {
                    for child in self.children.values() {
                        ans += child.approximate_range_counting(q, params);
                        // There is no need to know the exact count
                        if ans > 0 {
                            return ans;
                        }
                    }
                } else {
                    ans += self.cnt;
                }
            }
        }
        ans
    }
}

/// Gets the indexes of the intervals of the axes in the `D` dimensional space where lies a Cell with side
/// size equal to `side_size` that contains point `p`
pub fn get_cell_index<F: Float>(p: &ArrayView1<F>, side_size: F) -> Array1<i64> {
    let dimensionality = p.dim();
    let mut new_index = Array1::zeros(dimensionality);
    let half_size = side_size / F::from(2.0).unwrap();
    for (i, coord) in p.iter().enumerate() {
        if *coord >= (F::from(-1.0).unwrap() * half_size) && *coord < half_size {
            new_index[i] = 0;
        } else if *coord > F::from(0.0).unwrap() {
            new_index[i] = ((*coord - half_size) / side_size).ceil().to_i64().unwrap();
        } else {
            new_index[i] = -1 + ((*coord + half_size) / side_size).ceil().to_i64().unwrap();
        }
    }
    new_index
}

/// Gets the indexes of the intervals of the axes in the `D` dimensional space where lies a Cell with side
/// size equal to `epsilon/sqrt(D)` that contains point `p`
pub fn get_base_cell_index<F: Float>(
    p: &ArrayView1<F>,
    params: &AppxDbscanHyperParams<F>,
) -> Array1<i64> {
    let dimensionality = p.dim();
    get_cell_index(
        p,
        params.tolerance() / (F::from(dimensionality).unwrap()).sqrt(),
    )
}

/// Determines the type of intersection between a cell and an approximated ball.
/// The cell is determined by its center and the side of its size.
/// Returns:
///  * IntersectionType::FullyCovered if the cell is completely contained in a ball with center `q` and radius `epsilon(1 + rho)`;
///  * IntersectionType::Disjoint if the cell is completely outside of a ball with center `q` and radius `epsilon`;
///  * IntersectionType::Intersecting otherwise;
pub fn determine_intersection<F: Float>(
    q: &ArrayView1<F>,
    params: &AppxDbscanHyperParams<F>,
    cell_center: &Array1<F>,
    side_size: F,
) -> IntersectionType {
    let dimensionality = q.dim();

    //let appr_dist = (F::from(1.0).unwrap() + params.slack()) * params.tolerance();
    let dist_from_center = F::from(cell_center.l2_dist(&q).unwrap()).unwrap();
    let dist_corner_from_center =
        (side_size * F::from(dimensionality).unwrap().sqrt()) / F::from(2).unwrap();
    let min_dist_edge_from_center = side_size / F::from(2).unwrap();

    // sufficient condition to be disjoint: shortest possible distance from point q
    // to the closest point of the cell (lying on a corner) still greater than the tolerance
    if dist_from_center - dist_corner_from_center > params.tolerance() {
        // here we have sufficient and necessary conditions to be disjoint so we return that
        return IntersectionType::Disjoint;
    }

    // sufficient condition to be completely covered: distance from point to cell center plus distance from center to a corner
    // (farthest point from center) smaller or equal to the extended radius
    if dist_from_center + dist_corner_from_center <= params.appx_tolerance() {
        return IntersectionType::FullyCovered;
    }

    // sufficient condition to not be disjoint: longest possible distance from point q
    // to the closest point of the cell (lying on the middle of an 'egde') smaller than the tolerance
    if dist_from_center - min_dist_edge_from_center <= params.tolerance() {
        return IntersectionType::Intersecting;
    }

    let corners = get_corners(&cell_center, side_size);

    let mut farthest_corner_distance = F::zero();

    for corner in corners.axis_iter(Axis(0)) {
        let dist = F::from(corner.l2_dist(&q).unwrap()).unwrap();
        if dist > farthest_corner_distance {
            farthest_corner_distance = dist;
        }
    }

    if farthest_corner_distance < params.appx_tolerance() {
        return IntersectionType::FullyCovered;
    }
    return IntersectionType::Intersecting;
}

/// Gets the coordinates of all the corners (2^D) of a cell given its center points and its side size.
fn get_corners<F: Float>(cell_center: &Array1<F>, side_size: F) -> Array2<F> {
    let dist = side_size / F::from(2.0).unwrap();
    let dimensionality = cell_center.dim();
    //Ho 2^d combinazioni. Posso pensare ogni combinazione come un numero binario di d cifre.
    //Immagino di sostituire lo 0 con -dist e l'1 con +dist. Allora posso partire da cell_center
    //e fare la sua somma con ogni numero binario per trovare tutti i vertici
    let top = 2_usize.pow(dimensionality as u32);
    let mut corners = Array2::<F>::zeros((top, dimensionality));
    for bin_rep in 0..top {
        let mut new_corner = corners.row_mut(bin_rep);
        for bit_i in 0..dimensionality {
            let mask = 1 << bit_i;
            if bin_rep & mask == 0 {
                new_corner[bit_i] = cell_center[bit_i] - dist;
            } else {
                new_corner[bit_i] = cell_center[bit_i] + dist;
            }
        }
    }
    corners
}

fn cell_center_from_cell_index<F: Float>(cell_index: ArrayView1<i64>, side_size: F) -> Array1<F> {
    let dimensionality = cell_index.dim();
    let mut cell_center: Array1<F> = Array1::zeros(dimensionality);
    for i in 0..dimensionality {
        cell_center[i] = F::from(cell_index[i]).unwrap() * side_size;
    }
    cell_center
}

#[cfg(test)]
mod tests;
