use super::permutable_kernel::Permutable;
use super::{ExitReason, Float, Svm};

use ndarray::{Array1, Array2, ArrayView2, Axis};
#[cfg(feature = "serde")]
use serde_crate::{Deserialize, Serialize};
use std::marker::PhantomData;

/// Parameters of the solver routine
#[derive(Clone, Debug, PartialEq)]
pub struct SolverParams<F: Float> {
    /// Stopping condition
    pub eps: F,
    /// Should we shrink, e.g. ignore bounded alphas
    pub shrinking: bool,
}

/// Status of alpha variables of the solver
#[derive(Clone, Debug, PartialEq)]
struct Alpha<F: Float> {
    value: F,
    upper_bound: F,
}

impl<F: Float> Alpha<F> {
    pub fn from(value: F, upper_bound: F) -> Alpha<F> {
        Alpha { value, upper_bound }
    }

    pub fn reached_upper(&self) -> bool {
        self.value >= self.upper_bound
    }

    pub fn free_floating(&self) -> bool {
        self.value < self.upper_bound && self.value > F::zero()
    }

    pub fn reached_lower(&self) -> bool {
        self.value == F::zero()
    }

    pub fn val(&self) -> F {
        self.value
    }
}

#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
#[derive(Clone, Debug, PartialEq)]
pub enum SeparatingHyperplane<F: Float> {
    Linear(Array1<F>),
    WeightedCombination(Array2<F>),
}

/// Current state of the SMO solver
///
/// We are solving the dual problem with linear constraints
/// min_a f(a), s.t. y^Ta = d, 0 <= a_t < C, t = 1, ..., l
/// where f(a) = a^T Q a / 2 + p^T a
#[derive(Clone, Debug, PartialEq)]
pub struct SolverState<'a, F: Float, K: Permutable<F>> {
    /// Gradient of each variable
    gradient: Vec<F>,
    /// Cached gradient because most of the variables are constant
    gradient_fixed: Vec<F>,
    /// Current value of each variable and in respect to bounds
    alpha: Vec<Alpha<F>>,
    /// Active set of variables
    active_set: Vec<usize>,
    /// Number of active variables
    nactive: usize,
    unshrink: bool,
    nu_constraint: bool,
    r: F,

    /// Training data
    dataset: ArrayView2<'a, F>,

    /// Quadratic term of the problem
    kernel: K,
    /// Linear term of the problem
    p: Vec<F>,
    /// Targets we want to predict
    targets: Vec<bool>,
    /// Bounds per alpha
    bounds: Vec<F>,

    /// Parameters, e.g. stopping condition etc.
    params: SolverParams<F>,

    phantom: PhantomData<&'a K>,
}

#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
impl<'a, F: Float, K: 'a + Permutable<F>> SolverState<'a, F, K> {
    /// Initialize a solver state
    ///
    /// This is bounded by the lifetime of the kernel matrix, because it can quite large
    pub fn new(
        alpha: Vec<F>,
        p: Vec<F>,
        targets: Vec<bool>,
        dataset: ArrayView2<'a, F>,
        kernel: K,
        bounds: Vec<F>,
        params: SolverParams<F>,
        nu_constraint: bool,
    ) -> SolverState<'a, F, K> {
        // initialize alpha status according to bound
        let alpha = alpha
            .into_iter()
            .enumerate()
            .map(|(i, alpha)| Alpha::from(alpha, bounds[i]))
            .collect::<Vec<_>>();

        // initialize full active set
        let active_set = (0..alpha.len()).collect::<Vec<_>>();

        // initialize gradient
        let mut gradient = p.clone();
        let mut gradient_fixed = vec![F::zero(); alpha.len()];

        for i in 0..alpha.len() {
            // when we have reached alpha = F::zero(), then d(a) = p
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
            dataset,
            kernel,
            targets,
            bounds,
            params,
            nu_constraint,
            r: F::zero(),
            phantom: PhantomData,
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
    pub fn target(&self, idx: usize) -> F {
        if self.targets[idx] {
            F::one()
        } else {
            -F::one()
        }
    }

    /// Return the k-th bound
    pub fn bound(&self, idx: usize) -> F {
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
                + (F::one() + F::one()) * dist_i[j];
            if quad_coef <= F::zero() {
                quad_coef = F::cast(1e-10);
            }

            let delta = -(self.gradient[i] + self.gradient[j]) / quad_coef;
            let diff = self.alpha[i].val() - self.alpha[j].val();

            // update parameters
            self.alpha[i].value += delta;
            self.alpha[j].value += delta;

            // bound to feasible solution
            if diff > F::zero() {
                if self.alpha[j].val() < F::zero() {
                    self.alpha[j].value = F::zero();
                    self.alpha[i].value = diff;
                }
            } else if self.alpha[i].val() < F::zero() {
                self.alpha[i].value = F::zero();
                self.alpha[j].value = -diff;
            }

            if diff > bound_i - bound_j {
                if self.alpha[i].val() > bound_i {
                    self.alpha[i].value = bound_i;
                    self.alpha[j].value = bound_i - diff;
                }
            } else if self.alpha[j].val() > bound_j {
                self.alpha[j].value = bound_j;
                self.alpha[i].value = bound_j + diff;
            }
        } else {
            //dbg!(self.kernel.self_distance(i), self.kernel.self_distance(j), F::cast(2.0) * dist_i[j]);
            let mut quad_coef = self.kernel.self_distance(i) + self.kernel.self_distance(j)
                - F::cast(2.0) * dist_i[j];
            if quad_coef <= F::zero() {
                quad_coef = F::cast(1e-10);
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
            } else if self.alpha[j].val() < F::zero() {
                self.alpha[j].value = F::zero();
                self.alpha[i].value = sum;
            }
            if sum > bound_j {
                if self.alpha[j].val() > bound_j {
                    self.alpha[j].value = bound_j;
                    self.alpha[i].value = sum - bound_j;
                }
            } else if self.alpha[i].val() < F::zero() {
                self.alpha[i].value = F::zero();
                self.alpha[j].value = sum;
            }
            /*if self.alpha[i].val() > bound_i {
                self.alpha[i].value = bound_i;
            } else if self.alpha[i].val() < F::zero() {
                self.alpha[i].value = F::zero();
            }

            if self.alpha[j].val() > bound_j {
                self.alpha[j].value = bound_j;
            } else if self.alpha[j].val() < F::zero() {
                self.alpha[j].value = F::zero();
            }*/
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
    pub fn max_violating_pair(&self) -> ((F, isize), (F, isize)) {
        // max { -y_i * grad(f)_i \i in I_up(\alpha) }
        let mut gmax1 = (-F::infinity(), -1);
        // max { y_i * grad(f)_i \i in U_low(\alpha) }
        let mut gmax2 = (-F::infinity(), -1);

        for i in 0..self.nactive() {
            if self.targets[i] {
                if !self.alpha[i].reached_upper() && -self.gradient[i] >= gmax1.0 {
                    gmax1 = (-self.gradient[i], i as isize);
                }
                if !self.alpha[i].reached_lower() && self.gradient[i] >= gmax2.0 {
                    gmax2 = (self.gradient[i], i as isize);
                }
            } else {
                if !self.alpha[i].reached_upper() && -self.gradient[i] >= gmax2.0 {
                    gmax2 = (-self.gradient[i], i as isize);
                }
                if !self.alpha[i].reached_lower() && self.gradient[i] >= gmax1.0 {
                    gmax1 = (self.gradient[i], i as isize);
                }
            }
        }

        (gmax1, gmax2)
    }

    #[allow(clippy::type_complexity)]
    pub fn max_violating_pair_nu(&self) -> ((F, isize), (F, isize), (F, isize), (F, isize)) {
        let mut gmax1 = (-F::infinity(), -1);
        let mut gmax2 = (-F::infinity(), -1);
        let mut gmax3 = (-F::infinity(), -1);
        let mut gmax4 = (-F::infinity(), -1);

        for i in 0..self.nactive() {
            if self.targets[i] {
                if !self.alpha[i].reached_upper() && -self.gradient[i] > gmax1.0 {
                    gmax1 = (-self.gradient[i], i as isize);
                }
                if !self.alpha[i].reached_lower() && self.gradient[i] > gmax3.0 {
                    gmax3 = (self.gradient[i], i as isize);
                }
            } else {
                if !self.alpha[i].reached_upper() && -self.gradient[i] > gmax4.0 {
                    gmax4 = (-self.gradient[i], i as isize);
                }
                if !self.alpha[i].reached_lower() && self.gradient[i] > gmax2.0 {
                    gmax2 = (self.gradient[i], i as isize);
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

        let mut obj_diff_min = (F::infinity(), -1);

        if gmax.1 != -1 {
            let dist_i = self.kernel.distances(gmax.1 as usize, self.ntotal());

            for (j, dist_ij) in dist_i.into_iter().enumerate().take(self.nactive()) {
                if self.targets[j] {
                    if !self.alpha[j].reached_lower() {
                        let grad_diff = gmax.0 + self.gradient[j];
                        if grad_diff > F::zero() {
                            // this is possible, because op_i is some
                            let i = gmax.1 as usize;

                            let quad_coef = self.kernel.self_distance(i)
                                + self.kernel.self_distance(j)
                                - F::cast(2.0) * self.target(i) * dist_ij;

                            let obj_diff = if quad_coef > F::zero() {
                                -(grad_diff * grad_diff) / quad_coef
                            } else {
                                -(grad_diff * grad_diff) / F::cast(1e-10)
                            };

                            if obj_diff <= obj_diff_min.0 {
                                obj_diff_min = (obj_diff, j as isize);
                            }
                        }
                    }
                } else if !self.alpha[j].reached_upper() {
                    let grad_diff = gmax.0 - self.gradient[j];
                    if grad_diff > F::zero() {
                        // this is possible, because op_i is `Some`
                        let i = gmax.1 as usize;

                        let quad_coef = self.kernel.self_distance(i)
                            + self.kernel.self_distance(j)
                            + F::cast(2.0) * self.target(i) * dist_ij;

                        let obj_diff = if quad_coef > F::zero() {
                            -(grad_diff * grad_diff) / quad_coef
                        } else {
                            -(grad_diff * grad_diff) / F::cast(1e-10)
                        };
                        if obj_diff <= obj_diff_min.0 {
                            obj_diff_min = (obj_diff, j as isize);
                        }
                    }
                }
            }
        }

        if gmax.0 + gmax2.0 < self.params.eps || obj_diff_min.1 == -1 {
            (0, 0, true)
        } else {
            (gmax.1 as usize, obj_diff_min.1 as usize, false)
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

        let mut obj_diff_min = (F::infinity(), -1);

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
                    if grad_diff > F::zero() {
                        let dist_i_p = match dist_i_p {
                            Some(ref x) => x,
                            None => continue,
                        };

                        // this is possible, because op_i is some
                        let i = gmaxp1.1 as usize;

                        let quad_coef = self.kernel.self_distance(i) + self.kernel.self_distance(j)
                            - F::cast(2.0) * dist_i_p[j];

                        let obj_diff = if quad_coef > F::zero() {
                            -(grad_diff * grad_diff) / quad_coef
                        } else {
                            -(grad_diff * grad_diff) / F::cast(1e-10)
                        };

                        if obj_diff <= obj_diff_min.0 {
                            obj_diff_min = (obj_diff, j as isize);
                        }
                    }
                }
            } else if !self.alpha[j].reached_upper() {
                let grad_diff = gmaxn1.0 - self.gradient[j];
                if grad_diff > F::zero() {
                    let dist_i_n = match dist_i_n {
                        Some(ref x) => x,
                        None => continue,
                    };

                    // this is possible, because op_i is `Some`
                    let i = gmaxn1.1 as usize;

                    let quad_coef = self.kernel.self_distance(i) + self.kernel.self_distance(j)
                        - F::cast(2.0) * dist_i_n[j];

                    let obj_diff = if quad_coef > F::zero() {
                        -(grad_diff * grad_diff) / quad_coef
                    } else {
                        -(grad_diff * grad_diff) / F::cast(1e-10)
                    };
                    if obj_diff <= obj_diff_min.0 {
                        obj_diff_min = (obj_diff, j as isize);
                    }
                }
            }
        }

        if F::max(gmaxp1.0 + gmaxp2.0, gmaxn1.0 + gmaxn2.0) < self.params.eps
            || obj_diff_min.1 == -1
        {
            (0, 0, true)
        } else {
            let out_j = obj_diff_min.1 as usize;
            let out_i = if self.targets[out_j] {
                gmaxp1.1 as usize
            } else {
                gmaxn1.1 as usize
            };

            (out_i, out_j, false)
        }
    }

    pub fn should_shrunk(&self, i: usize, gmax1: F, gmax2: F) -> bool {
        if self.alpha[i].reached_upper() {
            if self.targets[i] {
                -self.gradient[i] > gmax1
            } else {
                -self.gradient[i] > gmax2
            }
        } else if self.alpha[i].reached_lower() {
            if self.targets[i] {
                self.gradient[i] > gmax2
            } else {
                -self.gradient[i] > gmax1
            }
        } else {
            false
        }
    }

    pub fn should_shrunk_nu(&self, i: usize, gmax1: F, gmax2: F, gmax3: F, gmax4: F) -> bool {
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
        if !self.unshrink && gmax1 + gmax2 <= self.params.eps * F::cast(10.0) {
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
        if !self.unshrink && F::max(gmax1 + gmax2, gmax3 + gmax4) <= self.params.eps * F::cast(10.0)
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

    pub fn calculate_rho(&mut self) -> F {
        // with additional constraint call the other function
        if self.nu_constraint {
            return self.calculate_rho_nu();
        }

        let mut nfree = 0;
        let mut sum_free = F::zero();
        let mut ub = F::infinity();
        let mut lb = -F::infinity();

        for i in 0..self.nactive() {
            let yg = self.target(i) * self.gradient[i];

            if self.alpha[i].reached_upper() {
                if self.targets[i] {
                    lb = F::max(lb, yg);
                } else {
                    ub = F::min(ub, yg);
                }
            } else if self.alpha[i].reached_lower() {
                if self.targets[i] {
                    ub = F::min(ub, yg);
                } else {
                    lb = F::max(lb, yg);
                }
            } else {
                nfree += 1;
                sum_free += yg;
            }
        }

        if nfree > 0 {
            sum_free / F::cast(nfree)
        } else {
            (ub + lb) / F::cast(2.0)
        }
    }

    pub fn calculate_rho_nu(&mut self) -> F {
        let (mut nfree1, mut nfree2) = (0, 0);
        let (mut sum_free1, mut sum_free2) = (F::zero(), F::zero());
        let (mut ub1, mut ub2) = (F::infinity(), F::infinity());
        let (mut lb1, mut lb2) = (-F::infinity(), -F::infinity());

        for i in 0..self.nactive() {
            if self.targets[i] {
                if self.alpha[i].reached_upper() {
                    lb1 = F::max(lb1, self.gradient[i]);
                } else if self.alpha[i].reached_lower() {
                    ub1 = F::max(ub1, self.gradient[i]);
                } else {
                    nfree1 += 1;
                    sum_free1 += self.gradient[i];
                }
            }

            if !self.targets[i] {
                if self.alpha[i].reached_upper() {
                    lb2 = F::max(lb2, self.gradient[i]);
                } else if self.alpha[i].reached_lower() {
                    ub2 = F::max(ub2, self.gradient[i]);
                } else {
                    nfree2 += 1;
                    sum_free2 += self.gradient[i];
                }
            }
        }

        let r1 = if nfree1 > 0 {
            sum_free1 / F::cast(nfree1)
        } else {
            (ub1 + lb1) / F::cast(2.0)
        };
        let r2 = if nfree2 > 0 {
            sum_free2 / F::cast(nfree2)
        } else {
            (ub2 + lb2) / F::cast(2.0)
        };

        self.r = (r1 + r2) / F::cast(2.0);

        (r1 - r2) / F::cast(2.0)
    }

    pub fn solve(mut self) -> Svm<F, F> {
        let mut iter = 0;
        let max_iter = if self.targets.len() > std::usize::MAX / 100 {
            std::usize::MAX
        } else {
            100 * self.targets.len()
        };

        let max_iter = usize::max(10_000_000, max_iter);
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

        if iter >= max_iter && self.nactive() < self.targets.len() {
            self.reconstruct_gradient();
            self.nactive = self.ntotal();
        }

        let rho = self.calculate_rho();
        let r = if self.nu_constraint {
            Some(self.r)
        } else {
            None
        };

        // calculate object function
        let mut v = F::zero();
        for i in 0..self.targets.len() {
            v += self.alpha[i].val() * (self.gradient[i] + self.p[i]);
        }
        let obj = v / F::cast(2.0);

        let exit_reason = if max_iter == iter {
            ExitReason::ReachedIterations
        } else {
            ExitReason::ReachedThreshold
        };

        // put back the solution
        let mut alpha: Vec<F> = (0..self.ntotal())
            .map(|i| self.alpha[self.active_set[i]].val())
            .collect();

        // If we are solving a regresssion problem the number of alpha values
        // computed by the solver are 2*(#samples). The final weights of each sample
        // is then computed as alpha[i] - alpha[#samples + i].
        // If instead the problem being solved is a calssification problem then
        // the alpha values are already in the same number as the samples and
        // they already represent their respective weights

        // Computing the final alpha vaues for regression
        if self.ntotal() > self.dataset.len_of(Axis(0)) {
            for i in 0..self.dataset.len_of(Axis(0)) {
                let tmp = alpha[i + self.dataset.len_of(Axis(0))];
                alpha[i] -= tmp;
            }
            alpha.truncate(self.dataset.len_of(Axis(0)));
        }

        // Make unmutable
        let alpha = alpha;

        // Now that the alpha values are set correctly we can proceed to calculate the
        // support vectors. If the kernel used is linear then they can be pre-combined
        // and we only need to store the vector given by their combination. If the kernel
        // is non linear then we need to store all support vectors so that we are able to
        // compute distances between them and new samples when making predictions.
        let sep_hyperplane = if self.kernel.inner().is_linear() {
            let mut tmp = Array1::zeros(self.dataset.len_of(Axis(1)));

            for (i, elm) in self.dataset.outer_iter().enumerate() {
                tmp.scaled_add(self.target(i) * alpha[i], &elm);
            }

            SeparatingHyperplane::Linear(tmp)
        } else {
            let support_vectors = self.dataset.select(
                Axis(0),
                &alpha
                    .iter()
                    .enumerate()
                    .filter(|(_, a)| a.abs() > F::cast(100.) * F::epsilon())
                    .map(|(i, _)| i)
                    .collect::<Vec<_>>(),
            );
            SeparatingHyperplane::WeightedCombination(support_vectors)
        };

        Svm {
            alpha,
            rho,
            r,
            exit_reason,
            obj,
            iterations: iter,
            sep_hyperplane,
            kernel_method: self.kernel.into_inner().method,
            probability_coeffs: None,
            phantom: PhantomData,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{SolverParams, SolverState};
    use crate::permutable_kernel::PermutableKernel;
    use crate::SeparatingHyperplane;

    #[test]
    fn autotraits() {
        fn has_autotraits<T: Send + Sync + Sized + Unpin>() {}
        has_autotraits::<SolverState<f64, PermutableKernel<f64>>>();
        has_autotraits::<SolverParams<f64>>();
        has_autotraits::<SeparatingHyperplane<f64>>();
    }
}
/*
/// Optimize the booth function
#[test]
fn test_booth_function() {
    let kernel = array![[10., 8.], [8., 10.]];
    let kernel = Kernel {
        inner: KernelInner::Dense(kernel.clone()),
        fnc: Box::new(|_,_| 0.0),
        dataset: &kernel
    };
    let targets = vec![true, true];
    let kernel = PermutableKernel::new(&kernel, targets.clone());

    let p = vec![-34., -38.];
    let params = SolverParams {
        eps: 1e-6,
        shrinking: false
    };

    let solver = SolverState::new(vec![1.0, 1.0], p, targets, kernel, vec![1000.0; 2], &params, false);

    let res: SvmBase<f64> = solver.solve();

    println!("{:?}", res.alpha);
    println!("{}", res);
}*/
