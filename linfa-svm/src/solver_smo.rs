use super::{ExitReason, Float, SolverParams, SvmResult};
use super::permutable_kernel::{Permutable, PermutableKernel, PermutableKernelOneClass, PermutableKernelRegression};
use linfa_kernel::Kernel;

/// Status of alpha variables of the solver
#[derive(Debug)]
struct Alpha<A: Float> {
    value: A,
    upper_bound: A,
}

impl<A: Float> Alpha<A> {
    pub fn from(value: A, upper_bound: A) -> Alpha<A> {
        Alpha {
            value,
            upper_bound,
        }
    }

    pub fn reached_upper(&self) -> bool {
        self.value >= self.upper_bound
    }

    pub fn free_floating(&self) -> bool {
        self.value < self.upper_bound && self.value > A::zero()
    }

    pub fn reached_lower(&self) -> bool {
        self.value == A::zero()
    }

    pub fn val(&self) -> A {
        self.value
    }
}

/// Current state of the SMO solver
///
/// We are solving the dual problem with linear constraint
/// min_a f(a), s.t. y^Ta = d, 0 <= a_t < C, t = 1, ..., l
/// where f(a) = a^T Q a / 2 + p^T a
pub struct SolverState<'a, A: Float, K: Permutable<'a, A>> {
    /// Gradient of each variable
    gradient: Vec<A>,
    /// Cached gradient because most of the variables are constant
    gradient_fixed: Vec<A>,
    /// Current value of each variable and in respect to bounds
    alpha: Vec<Alpha<A>>,
    /// Active set of variables
    active_set: Vec<usize>,
    /// Number of active variables
    nactive: usize,
    unshrink: bool,
    nu_constraint: bool,
    r: A,

    /// Quadratic term of the problem
    kernel: K,
    /// Linear term of the problem
    p: Vec<A>,
    /// Targets we want to predict
    targets: Vec<bool>,
    /// Bounds per alpha
    bounds: Vec<A>,

    /// Parameters, e.g. stopping condition etc.
    params: &'a SolverParams<A>,
}

impl<'a, A: Float, K: Permutable<'a, A>> SolverState<'a, A, K> {
    /// Initialize a solver state
    ///
    /// This is bounded by the lifetime of the kernel matrix, because it can quite large
    pub fn new(
        alpha: Vec<A>,
        p: Vec<A>,
        targets: Vec<bool>,
        kernel: K,
        bounds: Vec<A>,
        params: &'a SolverParams<A>,
        nu_constraint: bool,
    ) -> SolverState<'a, A, K> {
        // initialize alpha status according to bound
        let alpha = alpha
            .into_iter()
            .enumerate()
            .map(|(i, alpha)| Alpha::from(alpha, bounds[i]))
            .collect::<Vec<_>>();

        // initialize full active set
        let active_set = (0..alpha.len()).map(|i| i).collect::<Vec<_>>();

        // initialize gradient
        let mut gradient = p.clone();
        let mut gradient_fixed = vec![A::zero(); alpha.len()];

        for i in 0..alpha.len() {
            // when we have reached alpha = A::zero(), then d(a) = p
            if !alpha[i].reached_lower() {
                let dist_i = kernel.distances(i, alpha.len());
                let alpha_i = alpha[i].val();

                // update gradient as d(a) = p + Q a
                for j in 0..alpha.len() {
                    gradient[j] += alpha_i * dist_i[j];
                }

                // Cache gradient when we reached the upper bound for a variable
                if alpha[i].reached_upper() {
                    for j in 0..alpha.len() {
                        gradient_fixed[j] += bounds[i] * dist_i[j];
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
            kernel,
            targets,
            bounds,
            params,
            nu_constraint,
            r: A::zero(),
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
    pub fn target(&self, idx: usize) -> A {
        match self.targets[idx] {
            true => A::one(),
            false => -A::one(),
        }
    }

    /// Return the k-th bound
    pub fn bound(&self, idx: usize) -> A {
        self.bounds[idx]
    }

    /// Swap two variables
    pub fn swap(&mut self, i: usize, j: usize) {
        self.gradient.swap(i, j);
        self.gradient_fixed.swap(i, j);
        self.alpha.swap(i, j);
        self.p.swap(i, j);
        self.active_set.swap(i, j);
        self.kernel.swap_indices(i, j);
        self.targets.swap(i, j);
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

        let nfree: usize = (0..self.nactive())
            .filter(|x| self.alpha[*x].free_floating())
            .count();
        if nfree * self.ntotal() > 2 * self.nactive() * (self.ntotal() - self.nactive()) {
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

        let bound_i = self.bound(i);
        let bound_j = self.bound(j);

        let old_alpha_i = self.alpha[i].val();
        let old_alpha_j = self.alpha[j].val();

        if self.targets[i] != self.targets[j] {
            let mut quad_coef = self.kernel.self_distance(i)
                + self.kernel.self_distance(j)
                + (A::one() + A::one()) * dist_i[j];
            if quad_coef <= A::zero() {
                quad_coef = A::from(1e-10).unwrap();
            }

            let delta = -(self.gradient[i] + self.gradient[j]) / quad_coef;
            let diff = self.alpha[i].val() - self.alpha[j].val();

            // update parameters
            self.alpha[i].value += delta;
            self.alpha[j].value += delta;

            // bound to feasible solution
            if diff > A::zero() {
                if self.alpha[j].val() < A::zero() {
                    self.alpha[j].value = A::zero();
                    self.alpha[i].value = diff;
                }
            } else {
                if self.alpha[i].val() < A::zero() {
                    self.alpha[i].value = A::zero();
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
            let mut quad_coef = self.kernel.self_distance(i) + self.kernel.self_distance(j)
                - A::from(2.0).unwrap() * dist_i[j];
            if quad_coef <= A::zero() {
                quad_coef = A::from(1e-10).unwrap();
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
                if self.alpha[j].val() < A::zero() {
                    self.alpha[j].value = A::zero();
                    self.alpha[i].value = sum;
                }
            }
            if sum > bound_j {
                if self.alpha[j].val() > bound_j {
                    self.alpha[j].value = bound_j;
                    self.alpha[i].value = sum - bound_j;
                }
            } else {
                if self.alpha[i].val() < A::zero() {
                    self.alpha[i].value = A::zero();
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

        self.alpha[i] = Alpha::from(self.alpha[i].val(), self.bound(i));
        self.alpha[j] = Alpha::from(self.alpha[j].val(), self.bound(j));

        // update gradient of non-free variables if `i` became free or non-free
        if ui != self.alpha[i].reached_upper() {
            let dist_i = self.kernel.distances(i, self.ntotal());
            let bound_i = self.bound(i);
            if ui {
                for k in 0..self.ntotal() {
                    self.gradient_fixed[k] -= bound_i * dist_i[k];
                }
            } else {
                for k in 0..self.ntotal() {
                    self.gradient_fixed[k] += bound_i * dist_i[k];
                }
            }
        }

        // update gradient of non-free variables if `j` became free or non-free
        if uj != self.alpha[j].reached_upper() {
            let dist_j = self.kernel.distances(j, self.ntotal());
            let bound_j = self.bound(j);
            if uj {
                for k in 0..self.nactive() {
                    self.gradient_fixed[k] -= bound_j * dist_j[k];
                }
            } else {
                for k in 0..self.nactive() {
                    self.gradient_fixed[k] += bound_j * dist_j[k];
                }
            }
        }
    }

    /// Return max and min gradients of free variables
    pub fn max_violating_pair(&self) -> ((A, isize), (A, isize)) {
        // max { -y_i * grad(f)_i \i in I_up(\alpha) }
        let mut gmax1 = (-A::infinity(), -1);
        // max { y_i * grad(f)_i \i in U_low(\alpha) }
        let mut gmax2 = (-A::infinity(), -1);

        for i in 0..self.nactive() {
            if self.targets[i] {
                if !self.alpha[i].reached_upper() {
                    if -self.gradient[i] >= gmax1.0 {
                        gmax1 = (-self.gradient[i], i as isize);
                    }
                }
                if !self.alpha[i].reached_lower() {
                    if self.gradient[i] >= gmax2.0 {
                        gmax2 = (self.gradient[i], i as isize);
                    }
                }
            } else {
                if !self.alpha[i].reached_upper() {
                    if -self.gradient[i] >= gmax2.0 {
                        gmax2 = (-self.gradient[i], i as isize);
                    }
                }
                if !self.alpha[i].reached_lower() {
                    if self.gradient[i] >= gmax1.0 {
                        gmax1 = (self.gradient[i], i as isize);
                    }
                }
            }
        }

        (gmax1, gmax2)
    }

    pub fn max_violating_pair_nu(&self) -> ((A, isize), (A, isize), (A, isize), (A, isize)) {
        let mut gmax1 = (-A::infinity(), -1);
        let mut gmax2 = (-A::infinity(), -1);
        let mut gmax3 = (-A::infinity(), -1);
        let mut gmax4 = (-A::infinity(), -1);

        for i in 0..self.nactive() {
            if self.targets[i] {
                if !self.alpha[i].reached_upper() {
                    if -self.gradient[i] > gmax1.0 {
                        gmax1 = (-self.gradient[i], i as isize);
                    }
                }
                if !self.alpha[i].reached_lower() {
                    if self.gradient[i] > gmax3.0 {
                        gmax3 = (self.gradient[i], i as isize);
                    }
                }
            } else {
                if !self.alpha[i].reached_upper() {
                    if -self.gradient[i] > gmax4.0 {
                        gmax4 = (-self.gradient[i], i as isize);
                    }
                }
                if !self.alpha[i].reached_lower() {
                    if self.gradient[i] > gmax2.0 {
                        gmax2 = (self.gradient[i], i as isize);
                    }
                }
            }
        }

        (gmax1, gmax2, gmax3, gmax4)
    }

    /// Select optimal working set
    ///
    /// In each optimization step two variables are selected and then optimized. The indices are
    /// selected such that:
    ///  * i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    ///  * j: minimizes the decrease of the objective value
    pub fn select_working_set(&self) -> (usize, usize, bool) {
        if self.nu_constraint {
            return self.select_working_set_nu();
        }

        let (gmax, gmax2) = self.max_violating_pair();

        let mut obj_diff_min = (A::infinity(), -1);

        if gmax.1 != -1 {
            let dist_i = self.kernel.distances(gmax.1 as usize, self.ntotal());
            //dbg!(&dist_i, gmax, gmax2);

            for j in 0..self.nactive() {
                if self.targets[j] {
                    if !self.alpha[j].reached_lower() {
                        let grad_diff = gmax.0 + self.gradient[j];
                        if grad_diff > A::zero() {
                            // this is possible, because op_i is some
                            let i = gmax.1 as usize;

                            let quad_coef = self.kernel.self_distance(i)
                                + self.kernel.self_distance(j)
                                - A::from(2.0).unwrap() * self.target(i) * dist_i[j];

                            let obj_diff = if quad_coef > A::zero() {
                                -(grad_diff * grad_diff) / quad_coef
                            } else {
                                -(grad_diff * grad_diff) / A::from(1e-10).unwrap()
                            };

                            if obj_diff <= obj_diff_min.0 {
                                obj_diff_min = (obj_diff, j as isize);
                            }
                        }
                    }
                } else {
                    if !self.alpha[j].reached_upper() {
                        let grad_diff = gmax.0 - self.gradient[j];
                        if grad_diff > A::zero() {
                            // this is possible, because op_i is `Some`
                            let i = gmax.1 as usize;

                            let quad_coef = self.kernel.self_distance(i)
                                + self.kernel.self_distance(j)
                                + A::from(2.0).unwrap() * self.target(i) * dist_i[j];

                            let obj_diff = if quad_coef > A::zero() {
                                -(grad_diff * grad_diff) / quad_coef
                            } else {
                                -(grad_diff * grad_diff) / A::from(1e-10).unwrap()
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

    /// Select optimal working set
    ///
    /// In each optimization step two variables are selected and then optimized. The indices are
    /// selected such that:
    ///  * i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
    ///  * j: minimizes the decrease of the objective value
    pub fn select_working_set_nu(&self) -> (usize, usize, bool) {
        let (gmaxp1, gmaxn1, gmaxp2, gmaxn2) = self.max_violating_pair_nu();

        let mut obj_diff_min = (A::infinity(), -1);

        let dist_i_p = if gmaxp1.1 != -1 {
            Some(self.kernel.distances(gmaxp1.1 as usize, self.ntotal()))
        } else {
            None
        };

        let dist_i_n = if gmaxn1.1 != -1 {
            Some(self.kernel.distances(gmaxn1.1 as usize, self.ntotal()))
        } else {
            None
        };

        for j in 0..self.nactive() {
            if self.targets[j] {
                if !self.alpha[j].reached_lower() {
                    let grad_diff = gmaxp1.0 + self.gradient[j];
                    if grad_diff > A::zero() {
                        let dist_i_p = match dist_i_p {
                            Some(ref x) => x,
                            None => continue
                        };

                        // this is possible, because op_i is some
                        let i = gmaxp1.1 as usize;

                        let quad_coef = self.kernel.self_distance(i)
                            + self.kernel.self_distance(j)
                            - A::from(2.0).unwrap() * dist_i_p[j];

                        let obj_diff = if quad_coef > A::zero() {
                            -(grad_diff * grad_diff) / quad_coef
                        } else {
                            -(grad_diff * grad_diff) / A::from(1e-10).unwrap()
                        };

                        if obj_diff <= obj_diff_min.0 {
                            obj_diff_min = (obj_diff, j as isize);
                        }
                    }
                }
            } else {
                if !self.alpha[j].reached_upper() {
                    let grad_diff = gmaxn1.0 - self.gradient[j];
                    if grad_diff > A::zero() {
                        let dist_i_n = match dist_i_n {
                            Some(ref x) => x,
                            None => continue
                        };

                        // this is possible, because op_i is `Some`
                        let i = gmaxn1.1 as usize;

                        let quad_coef = self.kernel.self_distance(i)
                            + self.kernel.self_distance(j)
                            - A::from(2.0).unwrap() * dist_i_n[j];

                        let obj_diff = if quad_coef > A::zero() {
                            -(grad_diff * grad_diff) / quad_coef
                        } else {
                            -(grad_diff * grad_diff) / A::from(1e-10).unwrap()
                        };
                        if obj_diff <= obj_diff_min.0 {
                            obj_diff_min = (obj_diff, j as isize);
                        }
                    }
                }
            }
        }

        if A::max(gmaxp1.0, gmaxp2.0) + A::max(gmaxn1.0, gmaxn2.0) < self.params.eps || obj_diff_min.1 == -1 {
            return (0, 0, true);
        } else {
            let out_j = obj_diff_min.1 as usize;
            let out_i = if self.targets[out_j] {
                gmaxp1.1 as usize
            } else {
                gmaxn1.1 as usize
            };

            return (out_i, out_j, false);
        }
    }

    pub fn should_shrunk(&self, i: usize, gmax1: A, gmax2: A) -> bool {
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

    pub fn should_shrunk_nu(&self, i: usize, gmax1: A, gmax2: A, gmax3: A, gmax4: A) -> bool {
        if self.alpha[i].reached_upper() {
            if self.targets[i] {
                -self.gradient[i] > gmax1
            } else {
                -self.gradient[i] > gmax4
            }
        } else if self.alpha[i].reached_lower() {
            if self.targets[i] {
                self.gradient[i] > gmax2
            } else {
                self.gradient[i] > gmax3
            }
        } else {
            false
        }
    }

    pub fn do_shrinking(&mut self) {
        if self.nu_constraint {
            self.do_shrinking_nu();
            return;
        }

        let (gmax1, gmax2) = self.max_violating_pair();
        let (gmax1, gmax2) = (gmax1.0, gmax2.0);

        // work on all variables when 10*eps is reached
        if !self.unshrink && gmax1 + gmax2 <= self.params.eps * A::from(10.0).unwrap() {
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

    pub fn do_shrinking_nu(&mut self) {
        let (gmax1, gmax2, gmax3, gmax4) = self.max_violating_pair_nu();
        let (gmax1, gmax2, gmax3, gmax4) = (gmax1.0, gmax2.0, gmax3.0, gmax4.0);

        // work on all variables when 10*eps is reached
        if !self.unshrink
            && A::max(gmax1, gmax2) + A::max(gmax3, gmax4) <= self.params.eps * A::from(10.0).unwrap()
        {
            self.unshrink = true;
            self.reconstruct_gradient();
            self.nactive = self.ntotal();
        }

        // swap items until working set is homogeneous
        for i in 0..self.nactive() {
            if self.should_shrunk_nu(i, gmax1, gmax2, gmax3, gmax4) {
                self.nactive -= 1;
                // only consider items behing this one
                while self.nactive > i {
                    if !self.should_shrunk_nu(self.nactive(), gmax1, gmax2, gmax3, gmax4) {
                        self.swap(i, self.nactive());
                        break;
                    }
                    self.nactive -= 1;
                }
            }
        }
    }

    pub fn calculate_rho(&mut self) -> A {
        // with additional constraint call the other function
        if self.nu_constraint {
            return self.calculate_rho_nu();
        }

        let mut nfree = 0;
        let mut sum_free = A::zero();
        let mut ub = A::infinity();
        let mut lb = -A::infinity();

        for i in 0..self.nactive() {
            let yg = self.target(i) * self.gradient[i];

            if self.alpha[i].reached_upper() {
                if self.targets[i] {
                    lb = A::max(lb, yg);
                } else {
                    ub = A::min(ub, yg);
                }
            } else if self.alpha[i].reached_lower() {
                if self.targets[i] {
                    ub = A::min(ub, yg);
                } else {
                    lb = A::max(lb, yg);
                }
            } else {
                nfree += 1;
                sum_free += yg;
            }
        }

        if nfree > 0 {
            sum_free / A::from(nfree).unwrap()
        } else {
            (ub + lb) / A::from(2.0).unwrap()
        }
    }

    pub fn calculate_rho_nu(&mut self) -> A {
        let (mut nfree1, mut nfree2) = (0, 0);
        let (mut sum_free1, mut sum_free2) = (A::zero(), A::zero());
        let (mut ub1, mut ub2) = (A::infinity(), A::infinity());
        let (mut lb1, mut lb2) = (-A::infinity(), -A::infinity());

        for i in 0..self.nactive() {
            if self.targets[i] {
                if self.alpha[i].reached_upper() {
                    lb1 = A::max(lb1, self.gradient[i]);
                } else if self.alpha[i].reached_lower() {
                    ub1 = A::max(ub1, self.gradient[i]);
                } else {
                    nfree1 += 1;
                    sum_free1 += self.gradient[i];
                }
            } else {
                if self.alpha[i].reached_upper() {
                    lb2 = A::max(lb2, self.gradient[i]);
                } else if self.alpha[i].reached_lower() {
                    ub2 = A::max(ub2, self.gradient[i]);
                } else {
                    nfree2 += 1;
                    sum_free2 += self.gradient[i];
                }
            }
        }

        let r1 = if nfree1 > 0 {
            sum_free1 / A::from(nfree1).unwrap()
        } else {
            (ub1 + lb1) / A::from(2.0).unwrap()
        };
        let r2 = if nfree2 > 0 {
            sum_free2 / A::from(nfree2).unwrap()
        } else {
            (ub2 + lb2) / A::from(2.0).unwrap()
        };

        self.r = (r1+r2) / A::from(2.0).unwrap();

        (r1 - r2) / A::from(2.0).unwrap()
    }

    pub fn solve(mut self) -> SvmResult<'a, A> {
        let mut iter = 0;
        let max_iter = if self.targets.len() > std::usize::MAX / 100 {
            std::usize::MAX
        } else {
            100 * self.targets.len()
        };

        let max_iter = usize::max(10000000, max_iter);
        let mut counter = usize::min(self.targets.len(), 1000) + 1;
        while iter < max_iter {
            counter -= 1;
            if counter == 0 {
                counter = usize::min(self.ntotal(), 1000);
                if self.params.shrinking {
                    self.do_shrinking();
                }
            }

            let (mut i, mut j, is_optimal) = self.select_working_set();
            if is_optimal {
                self.reconstruct_gradient();
                let (i2, j2, is_optimal) = self.select_working_set();
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
            self.update((i, j));
        }

        if iter >= max_iter {
            if self.nactive() < self.targets.len() {
                self.reconstruct_gradient();
                self.nactive = self.ntotal();
            }
        }

        let rho = self.calculate_rho();
        let r = match self.nu_constraint {
            true => Some(self.r),
            false => None
        };

        // calculate object function
        let mut v = A::zero();
        for i in 0..self.targets.len() {
            v += self.alpha[i].val() * (self.gradient[i] + self.p[i]);
        }
        let obj = v / A::from(2.0).unwrap();

        let exit_reason = if max_iter == iter {
            ExitReason::ReachedIterations
        } else {
            ExitReason::ReachedThreshold
        };

        // put back the solution
        let alpha: Vec<A> = (0..self.targets.len())
            .map(|i| self.alpha[self.active_set[i]].val())
            .collect();

        SvmResult {
            alpha,
            rho,
            r,
            exit_reason,
            obj,
            iterations: iter,
            kernel: self.kernel.inner(),
        }
    }
}

pub struct Classification;

impl Classification {
    pub fn fit_c<'a, A: Float>(
        params: &'a SolverParams<A>,
        kernel: &'a Kernel<A>,
        targets: &'a [bool],
        cpos: A,
        cneg: A,
    ) -> SvmResult<'a, A> {
        let bounds = targets
            .iter()
            .map(|x| if *x { cpos } else { cneg })
            .collect::<Vec<_>>();

        let kernel = PermutableKernel::new(kernel, targets.to_vec());

        let solver = SolverState::new(
            vec![A::zero(); targets.len()],
            vec![-A::one(); targets.len()],
            targets.to_vec(),
            kernel,
            bounds,
            params,
            false,
        );

        let mut res = solver.solve();

        res.alpha = res
            .alpha
            .into_iter()
            .zip(targets.iter())
            .map(|(a, b)| if *b { a } else { -a })
            .collect();

        res
    }

    pub fn fit_nu<'a, A: Float>(
        params: &'a SolverParams<A>,
        kernel: &'a Kernel<A>,
        targets: &'a [bool],
        nu: A,
    ) -> SvmResult<'a, A> {
        let mut sum_pos = nu * A::from(targets.len()).unwrap() / A::from(2.0).unwrap();
        let mut sum_neg = nu * A::from(targets.len()).unwrap() / A::from(2.0).unwrap();
        let init_alpha = targets
            .iter()
            .map(|x| {
                if *x {
                    let val = A::min(A::one(), sum_pos);
                    sum_pos -= val;
                    val
                } else {
                    let val = A::min(A::one(), sum_neg);
                    sum_neg -= val;
                    val
                }
            })
            .collect::<Vec<_>>();

        let kernel = PermutableKernel::new(kernel, targets.to_vec());

        let solver = SolverState::new(
            init_alpha,
            vec![A::zero(); targets.len()],
            targets.to_vec(),
            kernel,
            vec![A::one(); targets.len()],
            params,
            true,
        );

        let mut res = solver.solve();

        let r = res.r.unwrap();

        res.alpha = res
            .alpha
            .into_iter()
            .zip(targets.iter())
            .map(|(a, b)| if *b { a } else { -a })
            .map(|x| x / r)
            .collect();
        res.rho /= r;
        res.obj /= r * r;

        res
    }

    pub fn fit_one_class<'a, A: Float + num_traits::ToPrimitive>(
        params: &'a SolverParams<A>,
        kernel: &'a Kernel<A>,
        nu: A,
    ) -> SvmResult<'a, A> {
        let size = kernel.size();
        let n = (nu * A::from(size).unwrap()).to_usize().unwrap();

        let init_alpha = (0..size)
            .map(|x| {
                if x < n {
                    A::one()
                } else if x == n {
                    nu * A::from(size).unwrap() - A::from(x).unwrap()
                } else {
                    A::zero()
                }
            })
            .collect::<Vec<_>>();

        let kernel = PermutableKernelOneClass::new(kernel);

        let solver = SolverState::new(
            init_alpha,
            vec![A::zero(); size],
            vec![true; size],
            kernel,
            vec![A::one(); size],
            params,
            false,
        );

        solver.solve()
    }
}

pub struct Regression;

impl Regression {
    pub fn fit_epsilon<'a, A: Float>(
        params: &'a SolverParams<A>,
        kernel: &'a Kernel<A>,
        target: &'a [A],
        c: A,
        p: A,
    ) -> SvmResult<'a, A> {
        let mut linear_term = vec![A::zero(); 2 * target.len()];
        let mut targets = vec![true; 2 * target.len()];

        for i in 0..target.len() {
            linear_term[i] = p - target[i];
            targets[i] = true;

            linear_term[i + target.len()] = p + target[i];
            targets[i] = false;
        }

        let kernel = PermutableKernelRegression::new(kernel);
        let solver = SolverState::new(
            vec![A::zero(); 2 * target.len()],
            linear_term,
            targets.to_vec(),
            kernel,
            vec![c; target.len()],
            params,
            false,
        );

        let mut res = solver.solve();

        for i in 0..target.len() {
            let tmp = res.alpha[i+target.len()];
            res.alpha[i] -= tmp;
        }
        res.alpha.truncate(target.len());

        res


    }

    pub fn fit_nu<'a, A: Float>(
        params: &'a SolverParams<A>,
        kernel: &'a Kernel<A>,
        target: &'a [A],
        c: A,
        nu: A,
    ) -> SvmResult<'a, A> {
        let mut alpha = vec![A::zero(); 2 * target.len()];
        let mut linear_term = vec![A::zero(); 2 * target.len()];
        let mut targets = vec![true; 2 * target.len()];

        let mut sum = c * nu * A::from(target.len()).unwrap() / A::from(2.0).unwrap();
        for i in 0..target.len() {
            alpha[i] = A::min(sum, c);
            alpha[i + target.len()] = A::min(sum, c);
            sum -= alpha[i];

            linear_term[i] = -target[i];
            targets[i] = true;

            linear_term[i + target.len()] = target[i];
            targets[i] = false;
        }

        let kernel = PermutableKernelRegression::new(kernel);
        let solver = SolverState::new(
            alpha,
            linear_term,
            targets.to_vec(),
            kernel,
            vec![c; target.len()],
            params,
            false,
        );

        solver.solve()
    }
}

#[cfg(test)]
mod tests {
    use super::{Classification, Regression, Permutable, PermutableKernel, SolverParams};
    use linfa::metrics::IntoConfusionMatrix;
    use linfa_kernel::Kernel;
    use ndarray::{array, Array, Axis};
    use ndarray_rand::rand_distr::Uniform;
    use ndarray_rand::RandomExt;

    #[test]
    fn test_swappable_kernel() {
        let dist = array![[1.0, 0.3, 0.1], [0.3, 1.0, 0.5], [0.1, 0.5, 1.0]];
        let targets = vec![true, true, true];
        let dist = Kernel::from_dense(dist);
        let mut kernel = PermutableKernel::new(&dist, targets);

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

    #[test]
    fn test_c_classification() {
        // generate two clusters with 100 samples each
        let entries = ndarray::stack(
            Axis(0),
            &[
                Array::random((10, 2), Uniform::new(-10., -5.)).view(),
                Array::random((10, 2), Uniform::new(5., 10.)).view(),
            ],
        )
        .unwrap();
        let targets = (0..20).map(|x| x < 10).collect::<Vec<_>>();

        let kernel = Kernel::gaussian(entries, 100.);
        let params = SolverParams {
            eps: 1e-3,
            shrinking: false,
        };

        let svc = Classification::fit_c(&params, &kernel, &targets, 1.0, 1.0);
        println!("{}", svc);

        let pred = kernel
            .dataset
            .outer_iter()
            .map(|x| svc.predict(x))
            .map(|x| x > 0.0)
            .collect::<Vec<_>>();

        let cm = pred.into_confusion_matrix(&targets);
        assert_eq!(cm.accuracy(), 1.0);
    }

    #[test]
    fn test_nu_classification() {
        // generate two clusters with 100 samples each
        let entries = ndarray::stack(
            Axis(0),
            &[
                Array::random((10, 2), Uniform::new(-10., -5.)).view(),
                Array::random((10, 2), Uniform::new(5., 10.)).view(),
            ],
        )
        .unwrap();
        let targets = (0..20).map(|x| x < 10).collect::<Vec<_>>();

        let kernel = Kernel::gaussian(entries, 100.);
        let params = SolverParams {
            eps: 1e-1,
            shrinking: false,
        };

        let svc = Classification::fit_nu(&params, &kernel, &targets, 0.1);
        println!("{}", svc);

        let pred = kernel
            .dataset
            .outer_iter()
            .map(|x| svc.predict(x))
            .map(|x| x > 0.0)
            .collect::<Vec<_>>();

        dbg!(&svc.alpha);
        let cm = pred.into_confusion_matrix(&targets);
        assert_eq!(cm.accuracy(), 1.0);
    }

    #[test]
    fn test_reject_classification() {
        // generate two clusters with 100 samples each
        let entries = Array::random((100, 2), Uniform::new(-4., 4.));
        let kernel = Kernel::gaussian(entries, 100.);

        let params = SolverParams {
            eps: 1e-3,
            shrinking: false,
        };

        let svc = Classification::fit_one_class(&params, &kernel, 0.1);
        println!("{}", svc);

        // now test that points outside the circle are rejected
        let validation = Array::random((100, 2), Uniform::new(-10., 10f32));
        let pred = validation
            .outer_iter()
            .map(|x| svc.predict(x) > 0.0)
            .collect::<Vec<_>>();

        // count the number of correctly rejected samples
        let mut rejected = 0;
        let mut total = 0;
        for (pred, pos) in pred.iter().zip(validation.outer_iter()) {
            let distance = (pos[0]*pos[0] + pos[1]*pos[1]).sqrt();
            if distance >= 5.0 {
                if !pred {
                    rejected += 1;
                }
                total += 1;
            }
        }

        // at least 95% should be correctly rejected
        assert!((rejected as f32) / (total as f32) > 0.95);
    }

    #[test]
    fn test_linear_epsilon_regression() {
        let target = Array::linspace(0., 10., 100);
        let entries = Array::ones((100, 2));
        let kernel = Kernel::gaussian(entries, 100.);

        let params = SolverParams {
            eps: 1e-3,
            shrinking: false
        };

        let svr = Regression::fit_epsilon(&params, &kernel, &target.as_slice().unwrap(), 1.0, 0.1);
        println!("{}", svr);
    }
}
