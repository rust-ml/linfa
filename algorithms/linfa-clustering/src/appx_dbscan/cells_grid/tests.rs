use crate::AppxDbscan;

use super::*;
use crate::appx_dbscan::cells_grid::cell::CoreCellInfo;
use linfa::prelude::ParamGuard;
use ndarray::{arr2, Array2};

#[test]
fn autotraits() {
    fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
    has_autotraits::<AppxDbscan>();
    has_autotraits::<StatusPoint>();
    has_autotraits::<CoreCellInfo<f64>>();
    has_autotraits::<Cell<f64>>();
}

#[test]
fn find_cells_test() {
    let params = AppxDbscan::params(2)
        .tolerance(2.0)
        .slack(0.1)
        .check()
        .unwrap();
    let l = params.tolerance / 2_f64.sqrt();
    let points = Array2::from_shape_vec((2, 2), vec![l, -l, -l, l]).unwrap();
    let grid = CellsGrid::new(points.view(), &params);
    assert_eq!(grid.cells().len(), 2);
}

#[test]
fn label_points_test() {
    let params = AppxDbscan::params(2)
        .tolerance(2.0)
        .slack(0.1)
        .check()
        .unwrap();
    let l = params.tolerance / 2_f64.sqrt();
    let all_points = vec![2.0 * l, 2.0 * l, 2.0 * l, 2.0 * l, 2.0 * l, 2.0 * l, l, l];
    let points = Array2::from_shape_vec((4, 2), all_points).unwrap();
    assert_eq!(points.row(0).dim(), 2);
    assert_eq!(points.nrows(), 4);
    let mut grid = CellsGrid::new(points.view(), &params);
    grid.label_points(points.view(), &params);
    assert_eq!(grid.cells().len(), 2);
    assert_eq!(grid.cells().iter().filter(|x| x.is_core()).count(), 2);
    assert_eq!(grid.cells().all_sets().count(), 1);
    for set in grid.cells().all_sets() {
        assert_eq!(set.count(), 2);
    }
    let all_points = vec![
        2.0 * l,
        2.0 * l,
        2.0 * l,
        2.0 * l,
        2.0 * l,
        2.0 * l,
        -5.0 * l,
        -5.0 * l,
    ];
    let points = Array2::from_shape_vec((4, 2), all_points).unwrap();
    let mut grid = CellsGrid::new(points.view(), &params);
    grid.label_points(points.view(), &params);
    assert_eq!(grid.cells().len(), 2);
    assert_eq!(grid.cells().iter().filter(|x| x.is_core()).count(), 1);
    assert_eq!(grid.cells.all_sets().count(), 2);
}

#[test]
fn populate_neighbours_test() {
    let tol = 2.0f64.sqrt();
    let params = AppxDbscan::params(4).tolerance(tol).check().unwrap();
    let points = arr2(&[
        [0., 0.],
        [0., 1.],
        [0., -1.],
        [1., 0.],
        [-1., 0.],
        [1., 1.],
        [-1., -1.],
        [1., -1.],
        [-1., 1.],
        [0., 2.0],
        [0., -2.0],
        [2.0, 0.],
        [-2.0, 0.],
        [1., 2.0],
        [1., -2.0],
        [-1., 2.0],
        [-1., -2.0],
        [2.0, 1.],
        [2.0, -1.],
        [-2.0, 1.],
        [-2.0, -1.],
        // These are not neighbors
        [2.1, 2.0],
        [2.0, -2.1],
        [-2.0, 2.1],
        [-2.1, -2.0],
    ]);

    let grid = CellsGrid::new(points.view(), &params);
    let origin_cell = &grid.cells()[0];
    let mut neighbours = origin_cell.neighbours_indexes().clone();
    neighbours.sort_unstable();
    // In 2D space, a cell should have 21 neighbour cells including itself, assuming all cells are
    // occupied.
    assert_eq!(neighbours, (0..=20).collect::<Vec<_>>());
}
