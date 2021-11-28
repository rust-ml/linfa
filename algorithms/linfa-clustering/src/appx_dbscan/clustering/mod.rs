use crate::appx_dbscan::cells_grid::CellsGrid;
use crate::appx_dbscan::AppxDbscanValidParams;
use linfa::Float;

use linfa_nn::NearestNeighbour;
use ndarray::{Array1, ArrayView2};

impl<F: Float, N: NearestNeighbour> AppxDbscanValidParams<F, N> {
    pub(crate) fn label(
        &self,
        grid: &mut CellsGrid<F>,
        points: ArrayView2<F>,
    ) -> Array1<Option<usize>> {
        let mut labels = self.label_connected_components(grid, points);
        self.label_border_noise_points(grid, points, &mut labels);
        labels
    }

    /// Explores the graph of cells contained in `grid` and labels all the core points of core cells in the same connected component
    /// with in the same cluster label, and core points from core cells in different connected components with different cluster labels.
    /// If the points in the input grid were not labeled then they will be inside this method.
    fn label_connected_components(
        &self,
        grid: &mut CellsGrid<F>,
        observations: ArrayView2<F>,
    ) -> Array1<Option<usize>> {
        if !grid.labeled() {
            grid.label_points(observations, self);
        }
        let mut labels = Array1::from_elem(observations.dim().0, None);
        let mut current_cluster_i: usize = 0;
        for set in grid.cells_mut().all_sets_mut() {
            let mut core_cells_count = 0;
            for cell in set.filter(|(_, c)| c.is_core()).map(|(_, c)| c) {
                cell.assign_to_cluster(current_cluster_i, &mut labels.view_mut());
                core_cells_count += 1;
            }
            if core_cells_count > 0 {
                current_cluster_i += 1;
            }
        }
        labels
    }

    /// Loops through all non core points of the dataset and labels them with one of the possible cluster labels that they belong to.
    /// If no such cluster is found, the point is given label of `None`.
    fn label_border_noise_points(
        &self,
        grid: &CellsGrid<F>,
        observations: ArrayView2<F>,
        clusters: &mut Array1<Option<usize>>,
    ) {
        for cell in grid.cells() {
            for cp_index in cell
                .points()
                .iter()
                .filter(|x| !x.is_core())
                .map(|x| x.index())
            {
                let curr_point = observations.row(cp_index);
                'nbrs: for neighbour_i in cell.neighbours_indexes() {
                    // indexes are added to neighbours only if tthey are in the table
                    let neighbour = grid.cells().get(*neighbour_i).unwrap();
                    if neighbour.approximate_range_counting(curr_point, self) > 0 {
                        clusters[cp_index] = Some(neighbour.cluster_i().unwrap_or_else(|| {
                            panic!("Attempted to get cluster index of a non core cell")
                        }));
                        // assign only to first matching cluster for compatibility
                        break 'nbrs;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
