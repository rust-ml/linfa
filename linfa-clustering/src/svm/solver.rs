use ndarray::Array1;
use linfa_kernel::Kernel;
use super::SvmResult;

/// Status of alpha variables of the solver
struct Alpha {
    value: f64,
    upper_bound: f64
}

impl Alpha {
    pub fn from(value: f64, upper_bound: f64) -> Alpha {
        Alpha {
            value,
            upper_bound
        }
    }

    pub fn reached_upper(&self) -> bool {
        self.value >= self.upper_bound
    }

    pub fn free_floating(&self) -> bool {
        self.value < self.upper_bound && self.value > 0.0
    }

    pub fn reached_lower(&self) -> bool {
        self.value > 0.0
    }

    pub fn val(&self) -> f64 {
        self.value
    }
}

/// Swappable kernel matrix
///
/// This struct wraps a kernel matrix with access indices. The working set can shrink during the
/// optimization and it is therefore necessary to reorder entries.
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

    /// Swap two indices
    pub fn swap_indices(&mut self, i: usize, j: usize) {
        self.kernel_indices.swap(i, j);
    }

    /// Return distances from node `idx` to all other nodes
    pub fn distances(&self, idx: usize, length: usize) -> Vec<f64> {
        let idx = self.kernel_indices[idx];

        let kernel = self.kernel.column(idx);

        // reorder entries
        (0..length)
            .into_iter()
            .map(|i| kernel[self.kernel_indices[i]])
            .collect()
    }

    /// Return distance to itself
    pub fn self_distance(&self, idx: usize) -> f64 {
        let idx = self.kernel_indices[idx];

        self.kernel_diag[idx]
    }
}

/// Parameters of the solver routine
#[derive(Clone)]
pub struct SolverParams {
    /// Stopping condition 
    eps: f64,
    /// Should we shrink, e.g. ignore bounded alphas
    shrinking: bool,
    /// Bounds for each alpha
    bounds: Vec<f64>,
}

impl SolverParams {
    /// Return the i-th bound
    pub fn bound(&self, idx: usize) -> f64 {
        self.bounds[idx]
    }
}

/// Current state of the SMO solver
///
/// We are solving the dual problem with linear constraint
/// min_a f(a), s.t. y^Ta = d, 0 <= a_t < C, t = 1, ..., l
/// where f(a) = a^T Q a / 2 + p^T a
pub struct SolverState<'a> {
    /// Gradient of each variable
    gradient: Vec<f64>,
    /// Cached gradient because most of the variables are constant
    gradient_fixed: Vec<f64>,
    /// Current value of each variable and in respect to bounds
    alpha: Vec<Alpha>,
    /// Active set of variables
    active_set: Vec<usize>,
    /// Number of active variables
    nactive: usize,
    unshrink: bool,

    /// Quadratic term of the problem
    kernel: KernelSwap<'a>,
    /// Linear term of the problem
    p: Vec<f64>,
    /// Targets we want to predict
    targets: &'a [bool],

    /// Parameters, e.g. stopping condition etc.
    params: SolverParams
}

impl<'a> SolverState<'a> {
    /// Initialize a solver state
    ///
    /// This is bounded by the lifetime of the kernel matrix, because it can quite large
    pub fn new(alpha: Vec<f64>, p: Vec<f64>, targets: &'a [bool], kernel: &'a Kernel<f64>, params: SolverParams) -> SolverState<'a> {

        // initialize alpha status according to bound
        let alpha = alpha.into_iter().enumerate()
            .map(|(i, alpha)| Alpha::from(alpha, params.bound(i)))
            .collect::<Vec<_>>();

        // initialize full active set
        let active_set = (0..alpha.len()).map(|i| i).collect::<Vec<_>>();

        // initialize gradient
        let mut gradient = p.clone();
        let mut gradient_fixed = vec![0.0; alpha.len()];

        for i in 0..alpha.len() {
            // when we have reached alpha = 0.0, then d(a) = p
            if !alpha[i].reached_lower() {
                let dist_i = kernel.column(i);
                let alpha_i = alpha[i].val();

                // update gradient as d(a) = p + Q a
                for j in 0..alpha.len() {
                    gradient[j] += alpha_i * dist_i[j];
                }

                // Cache gradient when we reached the upper bound for a variable
                if alpha[i].reached_upper() {
                    for j in 0..alpha.len() {
                        gradient_fixed[j] += params.bound(i) * dist_i[j];
                    }
                }
            }
        }

        SolverState {
            gradient,
            gradient_fixed,
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

    /// Return number of active variables
    pub fn nactive(&self) -> usize {
        self.nactive
    }

    /// Return number of total variables
    pub fn ntotal(&self) -> usize {
        self.alpha.len()
    }

    /// Return target as positive/negative indicator
    pub fn target(&self, idx: usize) -> f64 {
        match self.targets[idx] {
            true => 1.0,
            false => -1.0
        }
    }

    /// Swap two variables
    pub fn swap(&mut self, i: usize, j: usize) {
        self.gradient.swap(i, j);
        self.gradient_fixed.swap(i, j);
        self.alpha.swap(i, j);
        self.p.swap(i, j);
        self.active_set.swap(i, j);
        self.kernel.swap_indices(i, j);
    }

    /// Reconstruct gradients from inactivate variables
    ///
    /// A variables is inactive, when it reaches the upper bound. 
    ///
    fn reconstruct_gradient(&mut self) {
        // if no variable is inactive, skip
        if self.nactive() == self.ntotal() {
            return;
        }

        // d(a_i) = G^_i + p_i + ...
        for j in self.nactive()..self.ntotal() {
            self.gradient[j] = self.gradient_fixed[j] + self.p[j];
        }

        let nfree: usize = (0..self.nactive()).filter(|x| self.alpha[*x].free_floating()).count();
        if nfree*self.ntotal() > 2*self.nactive()*(self.ntotal()-self.nactive()){
            for i in self.nactive()..self.ntotal() {
                let dist_i = self.kernel.distances(i, self.nactive());
                for j in 0..self.nactive() {
                    if self.alpha[i].free_floating() {
                        self.gradient[i] += self.alpha[j].val() * dist_i[j];
                    }
                }
            }
        } else {
            for i in 0..self.nactive() {
                if self.alpha[i].free_floating() {
                    let dist_i = self.kernel.distances(i, self.ntotal());
                    let alpha_i = self.alpha[i].val();
                    for j in self.nactive()..self.ntotal() {
                        self.gradient[j] += alpha_i * dist_i[j];
                    }
                }
            }
        }
    }

    pub fn update(&mut self, working_set: (usize, usize)) {
        // working set indices are called i, j here
        let (i, j) = working_set;

        let dist_i = self.kernel.distances(i, self.nactive());
        let dist_j = self.kernel.distances(j, self.nactive());

        let bound_i = self.params.bound(i);
        let bound_j = self.params.bound(j);

        let old_alpha_i = self.alpha[i].val();
        let old_alpha_j = self.alpha[j].val();

        if self.targets[i] != self.targets[j] {
            let mut quad_coef = 
                self.kernel.self_distance(i) +
                self.kernel.self_distance(j) + 
                2.0 * dist_i[j];
            if quad_coef <= 0.0 {
                quad_coef = 1e-10;
            }

            let delta = -(self.gradient[i] + self.gradient[j]) / quad_coef;
            let diff = self.alpha[i].val() - self.alpha[j].val();

            // update parameters
            self.alpha[i].value += delta;
            self.alpha[j].value += delta;

            // bound to feasible solution
            if diff > 0.0 {
                if self.alpha[j].val() < 0.0 {
                    self.alpha[j].value = 0.0;
                    self.alpha[i].value = diff;
                } 
            } else {
                if self.alpha[i].val() < 0.0 {
                    self.alpha[i].value = 0.0;
                    self.alpha[j].value = -diff;
                }
            }
            if diff > bound_i - bound_j {
                if self.alpha[i].val() > bound_i {
                    self.alpha[i].value = bound_i;
                    self.alpha[j].value = bound_i - diff;
                }
            } else {
                if self.alpha[j].val() > bound_j {
                    self.alpha[j].value = bound_j;
                    self.alpha[i].value = bound_j + diff;
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

            let delta = (self.gradient[i] - self.gradient[j]) / quad_coef;
            let sum = self.alpha[i].val() + self.alpha[j].val();

            // update parameters
            self.alpha[i].value -= delta;
            self.alpha[j].value += delta;
            
            // bound to feasible solution
            if sum > bound_i {
                if self.alpha[i].val() > bound_i {
                    self.alpha[i].value = bound_i;
                    self.alpha[j].value = sum - bound_i;
                }
            } else {
                if self.alpha[j].val() < 0.0 {
                    self.alpha[j].value = 0.0;
                    self.alpha[i].value = sum;
                }
            }
            if sum > bound_j {
                if self.alpha[j].val() > bound_j {
                    self.alpha[j].value = bound_j;
                    self.alpha[i].value = sum - bound_j;
                }
            } else {
                if self.alpha[i].val() < 0.0 {
                    self.alpha[i].value = 0.0;
                    self.alpha[j].value = sum;
                }
            }
        }

        // update gradient
        let delta_alpha_i = self.alpha[i].val() - old_alpha_i;
        let delta_alpha_j = self.alpha[j].val() - old_alpha_j;

        for k in 0..self.nactive() {
            self.gradient[k] += dist_i[k] * delta_alpha_i + dist_j[k] * delta_alpha_j;
        }

        // update alpha status and gradient bar
        let ui = self.alpha[i].reached_upper();
        let uj = self.alpha[j].reached_upper();
        
        self.alpha[i] = Alpha::from(self.alpha[i].val(), self.params.bound(i));
        self.alpha[j] = Alpha::from(self.alpha[j].val(), self.params.bound(j));

        // update gradient of non-free variables if `i` became free or non-free
        if ui != self.alpha[i].reached_upper() {
            let dist_i = self.kernel.distances(i, self.ntotal());
            if ui {
                for k in 0..self.ntotal() {
                    self.gradient_fixed[k] -= self.params.bound(i) * dist_i[k];
                }
            } else {
                for k in 0..self.ntotal() {
                    self.gradient_fixed[k] += self.params.bound(i) * dist_i[k];
                }
            }
        }

        // update gradient of non-free variables if `j` became free or non-free
        if uj != self.alpha[j].reached_upper() {
            let dist_j = self.kernel.distances(j, self.ntotal());
            if uj {
                for k in 0..self.nactive() {
                    self.gradient_fixed[k] -= self.params.bound(j) * dist_j[k];
                }
            } else {
                for k in 0..self.nactive() {
                    self.gradient_fixed[k] += self.params.bound(j) * dist_j[k];
                }
            }
        }
    }

    /// Return max and min gradients of free variables
    pub fn max_violating_pair(&self) -> ((f64, isize), (f64, isize)) {
        // max { -y_i * grad(f)_i \i in I_up(\alpha) }
        let mut gmax1 = (-std::f64::INFINITY, -1);
        // max { y_i * grad(f)_i \i in U_low(\alpha) }
        let mut gmax2 = (-std::f64::INFINITY, -1);

        for i in 0..self.nactive() {
            if self.targets[i] {
                if self.alpha[i].reached_upper() {
                    if -self.gradient[i] >= gmax1.0 {
                        gmax1 = (-self.gradient[i], i as isize);
                    }
                }
                if self.alpha[i].reached_lower() {
                    if self.gradient[i] >= gmax2.0 {
                        gmax2 = (self.gradient[i], i as isize);
                    }
                }
            } else {
                if self.alpha[i].reached_upper() {
                    if -self.gradient[i] >= gmax2.0 {
                        gmax2 = (-self.gradient[i], i as isize);
                    }
                }
                if self.alpha[i].reached_lower() {
                    if self.gradient[i] >= gmax1.0 {
                        gmax1 = (self.gradient[i], i as isize);
                    }
                }
            }
        }

        (gmax1, gmax2)
    }

    /// Select optimal working set
    /// 
    /// In each optimization step two variables are selected and then optimized. The indices are
    /// selected such that:
    ///  * i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    ///  * j: minimizes the decrease of the objective value
    pub fn select_working_set(&self) -> (usize, usize, bool) {
        let (gmax, gmax2) = self.max_violating_pair();

        let mut obj_diff_min = (std::f64::INFINITY, -1);

        let dist_i = if gmax.1 != -1 {
            Some(self.kernel.distances(gmax.1 as usize, self.ntotal()))
        } else {
            None
        };

        for j in 0..self.nactive() {
            if self.targets[j] {
                if self.alpha[j].reached_lower() {
                    let grad_diff = gmax.0 + self.gradient[j];
                    if grad_diff > 0.0 {
                        if let Some(ref dist_i) = dist_i {
                            // this is possible, because op_i is some
                            let i = gmax.1 as usize;

                            let quad_coef = 
                                self.kernel.self_distance(i) +
                                self.kernel.self_distance(j) - 
                                2.0*self.target(i)*dist_i[j];

                            let obj_diff = if quad_coef > 0.0 {
                                -(grad_diff*grad_diff) / quad_coef
                            } else {
                                -(grad_diff*grad_diff) / 1e-10
                            };

                            if obj_diff <= obj_diff_min.0 {
                                obj_diff_min = (obj_diff, j as isize);
                            }
                        }
                    }
                }
            } else {
                if self.alpha[j].reached_upper() {
                    let grad_diff = gmax.0 - self.gradient[j];
                    if grad_diff > 0.0 {
                        if let Some(ref dist_i) = dist_i {
                            // this is possible, because op_i is `Some`
                            let i = gmax.1 as usize;

                            let quad_coef = 
                                self.kernel.self_distance(i) + 
                                self.kernel.self_distance(j) + 
                                2.0 * self.target(i) * dist_i[j];

                            let obj_diff = if quad_coef > 0.0 {
                                -(grad_diff*grad_diff) / quad_coef
                            } else {
                                -(grad_diff*grad_diff) / 1e-10
                            };
                            if obj_diff <= obj_diff_min.0 {
                                obj_diff_min = (obj_diff, j as isize);
                            }
                        }
                    }
                }
            }
        }

        if gmax.0 + gmax2.0 < self.params.eps || obj_diff_min.1 == -1 {
            return (0, 0, true);
        } else {
            return (gmax.1 as usize, obj_diff_min.1 as usize, false);
        }
    }

    pub fn should_shrunk(&self, i: usize, gmax1: f64, gmax2: f64) -> bool {
        if self.alpha[i].reached_upper() {
            if self.targets[i] {
                return -self.gradient[i] > gmax1;
            } else {
                return -self.gradient[i] > gmax2;
            }
        } else if self.alpha[i].reached_lower() {
            if self.targets[i] {
                return self.gradient[i] > gmax2;
            } else {
                return -self.gradient[i] > gmax1;
            }
        } else {
            return false;
        }
    }

    pub fn do_shrinking(&mut self) {
        let (gmax1, gmax2) = self.max_violating_pair();
        let (gmax1, gmax2) = (gmax1.0, gmax2.0);

        // work on all variables when 10*eps is reached
        if !self.unshrink && gmax1 + gmax2 <= self.params.eps * 10.0 {
            self.unshrink = true;
            self.reconstruct_gradient();
            self.nactive = self.ntotal();
        }

        // swap items until working set is homogeneous
        for i in 0..self.nactive() {
            if self.should_shrunk(i, gmax1, gmax2) {
                self.nactive -= 1;
                // only consider items behing this one
                while self.nactive > i {
                    if !self.should_shrunk(self.nactive(), gmax1, gmax2) {
                        self.swap(i, self.nactive());
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
            let yg = self.target(i) * self.gradient[i];

            if self.alpha[i].reached_upper() {
                if self.targets[i] {
                    lb = f64::max(lb, yg);
                } else {
                    ub = f64::min(ub, yg);
                }
            } else if self.alpha[i].reached_lower() {
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

    pub fn solve(&mut self) -> SvmResult {
        let mut status = SolverState::new(
            vec![0.0; self.targets.len()],
            vec![0.0; self.targets.len()],
            &self.targets,
            &self.kernel_matrix,
            self.params.clone()
        );

        let mut iter = 0;
        let max_iter = if self.targets.len() > std::usize::MAX / 100 {
            std::usize::MAX
        } else {
            100 * self.targets.len()
        };

        let max_iter = usize::max(10000000, max_iter);
        let mut counter = usize::min(self.targets.len(), 1000)+1;
        while iter < max_iter {
            counter -= 1;
            if counter == 0 {
                counter = usize::min(status.ntotal(), 1000);
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
            v += status.alpha[i].val() * (status.gradient[i] + status.p[i]);
        }
        let obj = v / 2.0;

        // put back the solution
        let alpha: Vec<f64> = (0..self.targets.len()).map(|i| status.alpha[status.active_set[i]].val()).collect();

        SvmResult {
            alpha,
            rho,
            obj
        }
    }
}

mod tests {
    use super::KernelSwap;
    use linfa_kernel::Kernel;
    use ndarray::{Array2, array};

    #[test]
    fn test_swappable_kernel() {
        let dist = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.5], [0.1, 0.5, 1.0]];
        let dist = Kernel::from_dense(dist);
        let mut kernel = KernelSwap::new(&dist);

        assert_eq!(kernel.distances(0, 3), &[1.0, 0.3, 0.1]);
        assert_eq!(kernel.distances(1, 3), &[0.3, 1.0, 0.5]);
        assert_eq!(kernel.distances(2, 3), &[0.1, 0.5, 1.0]);

        // swap first two nodes
        kernel.swap_indices(0, 1);

        assert_eq!(kernel.distances(0, 3), &[1.0, 0.3, 0.5]);
        assert_eq!(kernel.distances(1, 3), &[0.3, 1.0, 0.1]);
        assert_eq!(kernel.distances(2, 3), &[0.5, 0.1, 1.0]);

        // swap second and third node
        kernel.swap_indices(1, 2);

        assert_eq!(kernel.distances(0, 3), &[1.0, 0.5, 0.3]);
        assert_eq!(kernel.distances(1, 3), &[0.5, 1.0, 0.1]);
        assert_eq!(kernel.distances(2, 3), &[0.3, 0.1, 1.0]);

    }
}
