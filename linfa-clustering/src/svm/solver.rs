use ndarray::Array1;
use linfa_kernel::Kernel;

#[derive(PartialEq)]
enum AlphaStatus {
    LowerBound,
    UpperBound,
    Free
}

impl AlphaStatus {
    pub fn from(alpha: f64, bound: f64) -> AlphaStatus {
        if alpha >= bound {
            AlphaStatus::UpperBound
        } else if alpha <= 0.0 {
            AlphaStatus::LowerBound
        } else {
            AlphaStatus::Free
        }

    }
}

struct KernelSwap<'a> {
    kernel: &'a Kernel<f64>,
    kernel_diag: Array1<f64>,
    kernel_indices: Vec<usize>
}

impl<'a> KernelSwap<'a> {
    pub fn new(kernel: &'a Kernel<f64>) -> KernelSwap<'a> {
        let kernel_diag = kernel.diagonal();
        let kernel_indices = (0..kernel.size()).collect::<Vec<_>>();

        KernelSwap {
            kernel,
            kernel_diag,
            kernel_indices
        }
    }

    pub fn swap_indices(&mut self, i: usize, j: usize) {
        self.kernel_indices.swap(i, j);
    }

    pub fn distances(&self, idx: usize) -> Vec<f64> {
        let idx = self.kernel_indices[idx];

        self.kernel.get_row(idx)
    }

    pub fn self_distance(&self, idx: usize) -> f64 {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

#[derive(Clone)]
pub struct SolverParams {
    eps: f64,
    shrinking: bool,
    bounds: Vec<f64>,
}

impl SolverParams {
    pub fn bound(&self, idx: usize) -> f64 {
        self.bounds[idx]
    }
}

pub struct SolverState<'a> {
    gradient_objective: Vec<f64>,
    gradient_objective_hat: Vec<f64>,
    alpha: Vec<(f64, AlphaStatus)>,
    p: Vec<f64>,
    active_set: Vec<usize>,
    nactive: usize,
    unshrink: bool,

    targets: &'a [bool],
    kernel: KernelSwap<'a>,
    params: SolverParams
}

impl<'a> SolverState<'a> {
    pub fn new(alpha: Vec<f64>, p: Vec<f64>, targets: &'a [bool], kernel: &'a Kernel<f64>, params: SolverParams) -> SolverState<'a> {

        // initialize alpha status according to bound
        let alpha = alpha.into_iter().enumerate()
            .map(|(i, alpha)| (alpha, AlphaStatus::from(alpha, params.bound(i))))
            .collect::<Vec<_>>();

        // initialize full active set
        let active_set = (0..alpha.len()).map(|i| i).collect::<Vec<_>>();

        let mut gradient_objective = p.clone();
        let mut gradient_objective_hat = vec![0.0; alpha.len()];

        for i in 0..alpha.len() {
            if alpha[i].1 != AlphaStatus::LowerBound {
                let val = kernel.get_row(i);
                let alpha_i = alpha[i].0;
                for j in 0..alpha.len() {
                    gradient_objective[j] += alpha_i * val[i];
                }

                if alpha[i].1 != AlphaStatus::UpperBound {
                    for j in 0..alpha.len() {
                        gradient_objective_hat[j] += params.bound(i) * val[j];
                    }
                }
            }
        }

        SolverState {
            gradient_objective,
            gradient_objective_hat,
            alpha,
            p,
            nactive: active_set.len(),
            unshrink: false,
            active_set,
            targets,
            kernel: KernelSwap::new(kernel),
            params
        }
    }

    pub fn nactive(&self) -> usize {
        self.nactive
    }

    pub fn ntotal(&self) -> usize {
        self.alpha.len()
    }

    pub fn is_free(&self, i: usize) -> bool {
        self.alpha[i].1 == AlphaStatus::Free
    }

    pub fn swap_index(&mut self, i: usize, j: usize) {
        self.gradient_objective.swap(i, j);
        self.gradient_objective_hat.swap(i, j);
        self.alpha.swap(i, j);
        self.p.swap(i, j);
        self.active_set.swap(i, j);
        self.kernel.swap_indices(i, j);
    }

    /// Reconstruct gradients of inactivate elements
    fn reconstruct_gradient(&mut self) {
        if self.nactive() == self.ntotal() {
            return;
        }

        for j in self.nactive()..self.ntotal() {
            self.gradient_objective[j] = self.gradient_objective_hat[j] + self.p[j];
        }

        let nfree: usize = (0..self.nactive()).filter(|x| self.is_free(*x)).count();
        if 2*nfree < self.nactive() {
            println!("WARNING: usize -h 0 may be faster");
        }
        if nfree*self.ntotal() > 2*self.nactive()*(self.ntotal()-self.nactive()){
            for i in self.nactive()..self.ntotal() {
                let val = self.kernel.distances(i);
                for j in 0..self.nactive() {
                    if self.is_free(j) {
                        self.gradient_objective[i] += self.alpha[j].0 * val[j];
                    }
                }
            }
        } else {
            for i in 0..self.nactive() {
                if self.is_free(i) {
                    let val = self.kernel.distances(i);
                    let alpha_i = self.alpha[i].0;
                    for j in self.nactive()..self.ntotal() {
                        self.gradient_objective[j] += alpha_i * val[j];
                    }
                }
            }
        }
    }

    pub fn update(&mut self, working_set: (usize, usize)) {
        // working set indices are called i, j here
        let (i, j) = working_set;

        let dist_i = self.kernel.distances(i);
        let dist_j = self.kernel.distances(j);

        let bound_i = self.params.bound(i);
        let bound_j = self.params.bound(j);

        let old_alpha_i = self.alpha[i].0;
        let old_alpha_j = self.alpha[j].0;

        if self.targets[i] != self.targets[j] {
            let mut quad_coef = 
                self.kernel.self_distance(i) +
                self.kernel.self_distance(j) + 
                2.0 * dist_i[j];
            if quad_coef <= 0.0 {
                quad_coef = 1e-10;
            }

            let delta = -(self.gradient_objective[i] + self.gradient_objective[j]) / quad_coef;
            let diff = self.alpha[i].0 - self.alpha[j].0;
            self.alpha[i].0 += delta;
            self.alpha[j].0 += delta;

            if diff > 0.0 {
                if self.alpha[j].0 < 0.0 {
                    self.alpha[j].0 = 0.0;
                    self.alpha[i].0 = diff;
                } 
            } else {
                if self.alpha[i].0 < 0.0 {
                    self.alpha[i].0 = 0.0;
                    self.alpha[j].0 = -diff;
                }
            }
            if diff > bound_i - bound_j {
                if self.alpha[i].0 > bound_i {
                    self.alpha[i].0 = bound_i;
                    self.alpha[j].0 = bound_i - diff;
                }
            } else {
                if self.alpha[j].0 > bound_j {
                    self.alpha[j].0 = bound_j;
                    self.alpha[i].0 = bound_j + diff;
                }
            }
        } else {
            let mut quad_coef = 
                self.kernel.self_distance(i) +
                self.kernel.self_distance(j) -
                2.0 * dist_i[j];
            if quad_coef <= 0.0 {
                quad_coef = 1e-10;
            }

            let delta = (self.gradient_objective[i] - self.gradient_objective[j]) / quad_coef;
            let sum = self.alpha[i].0 + self.alpha[j].0;
            self.alpha[i].0 -= delta;
            self.alpha[j].0 += delta;
            
            if sum > bound_i {
                if self.alpha[i].0 > bound_i {
                    self.alpha[i].0 = bound_i;
                    self.alpha[j].0 = sum - bound_i;
                }
            } else {
                if self.alpha[j].0 < 0.0 {
                    self.alpha[j].0 = 0.0;
                    self.alpha[i].0 = sum;
                }
            }
            if sum > bound_j {
                if self.alpha[j].0 > bound_j {
                    self.alpha[j].0 = bound_j;
                    self.alpha[i].0 = sum - bound_j;
                }
            } else {
                if self.alpha[i].0 < 0.0 {
                    self.alpha[i].0 = 0.0;
                    self.alpha[j].0 = sum;
                }
            }
        }

        // update gradient
        let delta_alpha_i = self.alpha[i].0 - old_alpha_i;
        let delta_alpha_j = self.alpha[j].0 - old_alpha_j;

        for k in 0..self.nactive() {
            self.gradient_objective[k] += dist_i[k] * delta_alpha_i + dist_j[k] * delta_alpha_j;
        }

        // update alpha status and gradient bar
        let ui = self.alpha[i].1 == AlphaStatus::UpperBound;
        let uj = self.alpha[j].1 == AlphaStatus::UpperBound;
        
        self.alpha[i].1 = AlphaStatus::from(self.alpha[i].0, self.params.bound(i));
        self.alpha[j].1 = AlphaStatus::from(self.alpha[j].0, self.params.bound(j));

        if ui != (self.alpha[i].1 == AlphaStatus::UpperBound) {
            let dist_i = self.kernel.distances(i);
            if ui {
                for k in 0..self.nactive() {
                    self.gradient_objective_hat[k] -= self.params.bound(i) * dist_i[k];
                }
            } else {
                for k in 0..self.nactive() {
                    self.gradient_objective_hat[k] += self.params.bound(i) * dist_i[k];
                }
            }
        }

        if uj != (self.alpha[j].1 == AlphaStatus::UpperBound) {
            let dist_j = self.kernel.distances(j);
            if uj {
                for k in 0..self.nactive() {
                    self.gradient_objective_hat[k] -= self.params.bound(j) * dist_j[k];
                }
            } else {
                for k in 0..self.nactive() {
                    self.gradient_objective_hat[k] += self.params.bound(j) * dist_j[k];
                }
            }
        }
    }

    pub fn max_gradient(&self) -> ((f64, isize), (f64, isize)) {
        let mut gmax1 = (std::f64::INFINITY, -1);
        let mut gmax2 = (std::f64::INFINITY, -1);

        for i in 0..self.nactive() {
            if self.targets[i] {
                if self.alpha[i].1 != AlphaStatus::UpperBound {
                    if -self.gradient_objective[i] >= gmax1.0 {
                        gmax1.0 = -self.gradient_objective[i];
                    }
                }
                if self.alpha[i].1 != AlphaStatus::LowerBound {
                    if self.gradient_objective[i] >= gmax2.0 {
                        gmax2.0 = self.gradient_objective[i];
                    }
                }
            } else {
                if self.alpha[i].1 != AlphaStatus::UpperBound {
                    if -self.gradient_objective[i] >= gmax2.0 {
                        gmax2.0 = -self.gradient_objective[i];
                    }
                }
                if self.alpha[i].1 != AlphaStatus::LowerBound {
                    if self.gradient_objective[i] >= gmax1.0 {
                        gmax1.0 = self.gradient_objective[i];
                    }
                }
            }
        }

        (gmax1, gmax2)
    }

    pub fn select_working_set(&self) -> (usize, usize, bool) {
        let (gmax, gmax2) = self.max_gradient();

        let mut gmin_idx: isize = -1;
        let mut obj_diff_min = std::f64::INFINITY;
        let mut op_i = None;
        if gmax.1 != -1 {
            op_i = Some(self.kernel.distances(gmax.1 as usize));
        }

        for j in 0..self.nactive() {
            if self.targets[j] {
                if self.alpha[j].1 != AlphaStatus::LowerBound {
                    let grad_diff = gmax.0 + self.gradient_objective[j];
                    if grad_diff > 0.0 {
                        if let Some(ref op_i) = op_i {
                            // this is possible, because op_i is some
                            let i = gmax.1 as usize;

                            let quad_coeff = self.kernel.self_distance(i) +
                                self.kernel.self_distance(j) - 
                                2.0*(self.targets[i] as usize as f64)*op_i[j];

                            let obj_diff = if quad_coeff > 0.0 {
                                -(grad_diff*grad_diff) / quad_coeff
                            } else {
                                -(grad_diff*grad_diff) / 1e-10
                            };
                            if obj_diff <= obj_diff_min {
                                gmin_idx = j as isize;
                                obj_diff_min = obj_diff;
                            }
                        }
                    }
                }
            } else {
                if self.alpha[j].1 != AlphaStatus::UpperBound {
                    let grad_diff = gmax.0 - self.gradient_objective[j];
                    if grad_diff > 0.0 {
                        if let Some(ref op_i) = op_i {
                            // this is possible, because op_i is `Some`
                            let i = gmax.1 as usize;

                            let quad_coeff = self.kernel.self_distance(i) + 
                                self.kernel.self_distance(j) + 
                                2.0 * (self.targets[i] as usize as f64) * op_i[j];
                            let obj_diff = if quad_coeff > 0.0 {
                                -(grad_diff*grad_diff) / quad_coeff
                            } else {
                                -(grad_diff*grad_diff) / 1e-10
                            };
                            if obj_diff <= obj_diff_min {
                                gmin_idx = j as isize;
                                obj_diff_min = obj_diff;
                            }
                        }
                    }
                }
            }
        }

        if gmax.0 + gmax2.0 < self.params.eps || gmin_idx == -1 {
            return (0, 0, true);
        } else {
            return (gmax.1 as usize, gmin_idx as usize, false);
        }
    }

    pub fn should_shrunk(&self, i: usize, gmax1: f64, gmax2: f64) -> bool {
        if self.alpha[i].1 == AlphaStatus::UpperBound {
            if self.targets[i] {
                return -self.gradient_objective[i] > gmax1;
            } else {
                return -self.gradient_objective[i] > gmax2;
            }
        } else if self.alpha[i].1 == AlphaStatus::LowerBound {
            if self.targets[i] {
                return self.gradient_objective[i] > gmax2;
            } else {
                return -self.gradient_objective[i] > gmax1;
            }
        } else {
            return false;
        }
    }

    pub fn do_shrinking(&mut self) {
        let (gmax1, gmax2) = self.max_gradient();
        let (gmax1, gmax2) = (gmax1.0, gmax2.0);

        if self.unshrink && gmax1 + gmax2 <= self.params.eps * 10.0 {
            self.unshrink = true;
            self.reconstruct_gradient();
            self.nactive = self.ntotal();
        }

        for i in 0..self.nactive() {
            if self.should_shrunk(i, gmax1, gmax2) {
                self.nactive -= 1;
                while self.nactive > i {
                    if !self.should_shrunk(self.nactive, gmax1, gmax2) {
                        self.swap_index(i, self.nactive());
                        break;
                    }
                    self.nactive -= 1;
                }
            }
        }
    }

    pub fn calculate_rho(&self) -> f64 {
        let mut nfree = 0;
        let mut sum_free = 0.0;
        let mut ub = std::f64::INFINITY;
        let mut lb = -std::f64::INFINITY;

        for i in 0..self.nactive() {
            let yg = (self.targets[i] as usize as f64) * self.gradient_objective[i];

            if self.alpha[i].1 == AlphaStatus::UpperBound {
                if self.targets[i] {
                    lb = f64::max(lb, yg);
                } else {
                    ub = f64::min(ub, yg);
                }
            } else if self.alpha[i].1 == AlphaStatus::LowerBound {
                if self.targets[i] {
                    ub = f64::min(ub, yg);
                } else {
                    lb = f64::max(lb, yg);
                }
            } else {
                nfree += 1;
                sum_free += yg;
            }
        }

        if nfree > 0 {
            sum_free / nfree as f64
        } else {
            (ub+lb) / 2.0
        }
    }
}


pub struct Solver {
    params: SolverParams,
    /// dataset
    kernel_matrix: Kernel<f64>,
    targets: Vec<bool>,
}

impl Solver {
    pub fn new(params: SolverParams, kernel_matrix: Kernel<f64>, targets: Vec<bool>) -> Solver {
        Solver {
            params,
            kernel_matrix,
            targets
        }
    }

    pub fn solve(&mut self) {
        let mut status = SolverState::new(
            vec![0.0; self.targets.len()],
            vec![0.0; self.targets.len()],
            &self.targets,
            &self.kernel_matrix,
            self.params.clone()
        );

        let mut iter = 0;
        let max_iter = usize::max(10000000, 100*self.targets.len());
        let mut counter = usize::min(self.targets.len(), 1000)+1;
        while iter < max_iter {
            counter -= 1;
            if counter == 0 {
                counter = usize::min(status.nactive(), 1000);
                if self.params.shrinking {
                    status.do_shrinking();
                }
            }

            let (mut i, mut j, is_optimal) = status.select_working_set();
            if is_optimal {
                status.reconstruct_gradient();
                let (i2, j2, is_optimal) = status.select_working_set();
                if is_optimal {
                    break;
                } else {
                    // do shrinking next iteration
                    counter = 1;
                    i = i2;
                    j = j2;
                }
            }

            iter += 1;

            // update alpha[i] and alpha[j]
            status.update((i, j));
        }

        if iter >= max_iter {
            if status.nactive() < self.targets.len() {
                status.reconstruct_gradient();
                status.nactive = status.ntotal();
            }
        }

        let rho = status.calculate_rho();

        // calculate object function
        let mut v = 0.0;
        for i in 0..self.targets.len() {
            v += status.alpha[i].0 * (status.gradient_objective[i] + status.p[i]);
        }
        let obj = v / 2.0;

        // put back the solution
        let alpha: Vec<usize> = (0..self.targets.len()).map(|i| status.active_set[i]).collect();
    }
}
