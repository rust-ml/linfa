mod cell;

use std::collections::HashMap;

use crate::appx_dbscan::counting_tree::get_base_cell_index;
use crate::AppxDbscanValidParams;
use linfa::Float;
use linfa_nn::{distance::L2Dist, NearestNeighbour};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use partitions::PartitionVec;

use cell::{Cell, StatusPoint};

pub type CellVector<F> = PartitionVec<Cell<F>>;
/// A structure that memorizes all non empty cells by their index's hash
pub type CellTable = HashMap<Array1<i64>, usize>;

#[derive(Debug, Clone, PartialEq)]
pub struct CellsGrid<F: Float> {
    table: CellTable,
    cells: CellVector<F>,
    labeled: bool,
    dimensionality: usize,
}

impl<F: Float> CellsGrid<F> {
    /// Partitions the euclidean space containing `points` in a grid
    pub fn new<N: NearestNeighbour>(
        points: ArrayView2<F>,
        params: &AppxDbscanValidParams<F, N>,
    ) -> CellsGrid<F> {
        let mut grid = CellsGrid {
            table: CellTable::with_capacity(points.dim().0),
            cells: PartitionVec::with_capacity(points.dim().0),
            dimensionality: points.ncols(),
            labeled: false,
        };
        grid.populate(points, params);
        grid
    }

    /// Divides the D dimensional euclidean space in a grid of cells with side length `epsilon\sqrt(D)` and memorizes
    /// the non empty ones in a `CellTable`
    pub fn populate<N: NearestNeighbour>(
        &mut self,
        points: ArrayView2<F>,
        params: &AppxDbscanValidParams<F, N>,
    ) {
        for (p_i, curr_point) in points.axis_iter(Axis(0)).enumerate() {
            self.insert_point(curr_point, p_i, params);
        }
        self.populate_neighbours(params);
    }

    /// Function that decides which points from each cell are core points and which cells are core cells.
    ///
    /// # Arguments:
    ///
    /// * `points`: The points to cluster
    /// * `params`: the DBSCAN algorithm parameters
    ///
    /// # Behavior
    /// At the end of this function all the possible unions between neighbouring cells in `self.cells` will have been made.
    pub fn label_points<N>(&mut self, points: ArrayView2<F>, params: &AppxDbscanValidParams<F, N>) {
        for cell_i in self.table.values() {
            let mut cloned_cell = self.cells[*cell_i].clone();
            cloned_cell.label(&self.cells, points, params);
            self.cells[*cell_i] = cloned_cell;
        }
        self.unite_neighbouring_cells(points, params);
        self.labeled = true;
    }

    fn populate_neighbours<N: NearestNeighbour>(&mut self, params: &AppxDbscanValidParams<F, N>) {
        let nindices = self.cells.len();
        // populate the array with the indices of all cells
        let mut indices = Array2::zeros((nindices, self.dimensionality));
        for (cell, mut index) in self.cells.iter().zip(indices.rows_mut()) {
            index.assign(&cell.index);
        }

        // Neighbour cells are cells that are within `tolerance` of any edge of the cell.
        // Finding neighbour cells via range query is iffy because the cell indices are center
        // coordinates, not edge coordinates. This means we need a range above `tolerance` to
        // detect cells whose edges are within `tolerance` of the cell's edge.
        // The current range of `2*tolerance` detects all neighbour cells as well as some extra
        // cells. Lowering the range improves performance but can adversely affect accuracy, since
        // neighbours cells may be missed.
        let range = params.tolerance * F::cast(2);

        // bulk load the NN structure with all cell indices that are actually in the table
        let nn = params
            .nn_algo()
            .from_batch(&indices, L2Dist)
            .expect("nearest neighbour initialization should not fail");
        for cell in self.cells.iter_mut() {
            let spatial_index = cell.index.view();
            let neighbors = nn.within_range(spatial_index, range).unwrap();
            let neighbors = neighbors.into_iter().map(|(_, i)| i).collect();
            cell.populate_neighbours(neighbors);
        }
    }

    pub fn labeled(&self) -> &bool {
        &self.labeled
    }

    pub fn cells(&self) -> &PartitionVec<Cell<F>> {
        &self.cells
    }

    pub fn cells_mut(&mut self) -> &mut PartitionVec<Cell<F>> {
        &mut self.cells
    }

    fn insert_point<N>(
        &mut self,
        point: ArrayView1<F>,
        p_i: usize,
        params: &AppxDbscanValidParams<F, N>,
    ) {
        let cell_index = get_base_cell_index(point, params);
        let curr_cell_n = self.cells.len();
        let cell_i = self.table.entry(cell_index).or_insert(curr_cell_n);
        if *cell_i == curr_cell_n {
            self.cells.push(Cell::new(point.to_owned()));
        }
        self.cells[*cell_i].points_mut().push(StatusPoint::new(p_i));
    }

    fn unite_neighbouring_cells<N>(
        &mut self,
        points: ArrayView2<F>,
        params: &AppxDbscanValidParams<F, N>,
    ) {
        for cell_i in self.table.values() {
            if !self.cells[*cell_i].is_core() {
                continue;
            }
            let curr_cell_points = self.cells[*cell_i].points().clone();
            let neighbours_indexes = self.cells[*cell_i].neighbours_indexes().clone();
            for n_index in neighbours_indexes {
                let neighbour = self.cells.get(n_index).unwrap();
                if !neighbour.is_core() || self.cells.same_set(*cell_i, n_index) {
                    continue;
                }
                for point in curr_cell_points.iter().filter(|p| p.is_core()) {
                    if neighbour.approximate_range_counting(points.row(point.index()), params) > 0 {
                        self.cells.union(*cell_i, n_index);
                        break;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
