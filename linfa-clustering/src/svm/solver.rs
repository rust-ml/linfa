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

pub struct SolverParams {
    
}

impl SolverParams {
    pub fn get_c(&self, idx: usize) -> f64 {
        0.0
    }
}

pub struct SolverState<'a> {
    gradient_objective: Vec<f64>,
    gradient_objective_hat: Vec<f64>,
    alpha: Vec<(f64, AlphaStatus)>,
    p: Vec<f64>,
    active_set: Vec<usize>,
    nactive: usize,

    kernel: &'a Kernel<f64>,
    params: SolverParams
}

impl<'a> SolverState<'a> {
    pub fn new(alpha: Vec<f64>, p: Vec<f64>, kernel: &'a Kernel<f64>, params: SolverParams) -> SolverState<'a> {

        // initialize alpha status
        let alpha = alpha.into_iter().enumerate()
            .map(|(i, alpha)| (alpha, AlphaStatus::from(alpha, params.get_c(i))))
            .collect::<Vec<_>>();

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
                        gradient_objective_hat[j] += params.get_c(i) * val[j];
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
            active_set,
            kernel,
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
    }

    fn reconstruct_gradient(&mut self) {
        if self.nactive() == self.ntotal() {
            return;
        }

        for j in self.nactive()..self.ntotal() {
            self.gradient_objective[j] = self.gradient_objective_hat[j] + self.p[j];
        }

        let nfree: usize = (0..self.nactive()).filter(|x| self.is_free(*x)).sum();
        if 2*nfree < self.nactive() {
            println!("WARNING: usize -h 0 may be faster");
        }
        if nfree*self.ntotal() > 2*self.nactive()*(self.ntotal()-self.nactive()){
            for i in self.nactive()..self.ntotal() {
                let val = self.kernel.get_row(i);
                for j in 0..self.nactive() {
                    if self.is_free(j) {
                        self.gradient_objective[i] += self.alpha[j].0 * val[j];
                    }
                }
            }
        } else {
            for i in 0..self.nactive() {
                if self.is_free(i) {
                    let val = self.kernel.get_row(i);
                    let alpha_i = self.alpha[i].0;
                    for j in self.nactive()..self.ntotal() {
                        self.gradient_objective[j] += alpha_i * val[j];
                    }
                }
            }
        }
    }

}

    /*
    pub fn max_gradient(&self, classes: &[bool]) -> ((f64, isize), (f64, isize)) {
        let mut gmax = (f64::INFINITY, -1);
        let mut gmax2 = (f64::INFINITY, -1);

        for i in 0..self.nactive() {
            if classes[i] {
                if self.alpha[i].1 != AlphaStatus::UpperBound {
                    if -self.gradient_objective[i] >= gmax1 {
                        gmax1 = -self.gradient_objective[i];
                    }
                }
                if self.alpha[i].1 != AlphaStatus::LowerBound {
                    if self.gradient_objective[i] >= gmax2 {
                        gmax2 = self.gradient_objective[i];
                    }
                }
            } else {
                if self.alpha[i].1 != AlphaStatus::UpperBound {
                    if -self.gradient_objective[i] >= gmax2 {
                        gmax2 = -self.gradient_objective[i];
                    }
                }
                if self.alpha[i].1 != AlphaStatus::LowerBound {
                    if self.gradient_objective[i] >= gmax1 {
                        gmax1 = self.gradient_objective[i];
                    }
                }
            }
        }

        (gmax, gmax2)
    }

    pub fn select_working_set(&self, kernel_matrix: &Kernel, classes: &[bool]) -> (usize, usize, bool) {
        let (gmax, gmax2) = self.max_gradient(classes);

        let mut gmin_idx = -1;
        let mut obj_diff_min = f64::INFINITY;
        let mut op_i = None;
        if gmax.0 != -1 {
            op_i = kernel_matrix.get(gmax.0, self.nactive());
        }

        for j in 0..self.nactive() {
            if classes[j] {
                if self.alpha[j] != AlphaStatus::LowerBound {
                    let grad_diff = gmax.0 + self.gradient_objective[j];
                    if grad_diff > 0.0 {
                        let quad_coeff = op_d[i]*op_d[j]-2.0*classes[i]*op_i[j];
                        let obj_diff = if quad_coeff > 0.0 {
                            -(grad_diff*grad_diff) / quad_coeff
                        } else {
                            -(grad_diff*grad_diff) / 1e-10
                        };
                        if obj_diff <= obj_diff_min {
                            gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            } else {
                if self.alpha[j] != AlphaStatus::UpperBound {
                    let grad_diff = gmax.0 - self.gradient_objective[j];
                    if grad_diff > 0.0 {
                        let quad_coeff = op_d[i] + op_d[j] + 2.0 * classes[i] * op_i[j];
                        let obj_diff = if quad_coeff > 0.0 {
                            -(grad_diff*grad_diff) / quad_coeff
                        } else {
                            -(grad_dif*grad_diff) / 1e-10
                        };
                        if obj_diff <= obj_diff_min {
                            gmin_idx = j;
                            obj_diff_min = obj_diff;
                        }
                    }
                }
            }
        }

        if gmax.0 + gmax2.0 < params.eps || gmin_idx == -1 {
            return (0, 0, true);
        } else {
            return (gmax.0, gmin_idx, false);
        }
    }

    pub fn should_shrunk(&self, i: usize, gmax1: usize, gmax2: usize, classes: &[bool]) -> bool {
        if self.alpha[i].1 == AlphaStatus.UpperBound {
            if classes[i] {
                return -self.gradient_objective[i] > gmax1;
            } else {
                return -self.gradient_objective[i] > gmax2;
            }
        } else if self.alpha[i].1 == AlphaStatus.LowerBound {
            if classes[i] {
                return self.gradient_objective[i] > gmax2;
            } else {
                return -self.gradient_objective[i] > gmax1;
            }
        } else {
            return false;
        }
    }

    pub fn calculate_rho(&self, classes: &[bool]) -> f64 {
        let mut nfree = 0;
        let mut sum_free = 0.0;
        let mut ub = f64::INFINITY;
        let mut lb = -f64::INFINITY;

        for i in 0..self.nactive() {
            let yG = classes[i] * self.gradient_objective[i];

            if self.alpha[i].1 == AlphaStatus::UpperBound {
                if classes[i] {
                    lb = max(lb, yG);
                } else {
                    ub = min(ub, yG);
                }
            } else if self.alpha[i].1 == AlphaStatus::LowerBound {
                if classes[i] {
                    ub = min(ub, yG);
                } else {
                    lb = max(lb, yG);
                }
            } else {
                nfree += 1;
                sum_free += yG;
            }
        }

        if nfree > 0 {
            sum_free / nfree
        } else {
            (ub+lb) / 2.0
        }
    }
}


pub struct Solver {
    params: SolverParams,
    /// dataset
    kernel_matrix: Kernel,
    class: Vec<bool>,
}

impl Solver {
    pub fn new(params: &SolverParams, kernel_matrix: Kernel, class: Vec<bool>) -> Solver {
        Solver {
            params,
            kernel_matrix,
            class
        }
    }

    pub fn get_c(i: usize) -> f64 {
        if class[i] {
            self.params.c_pos
        } else {
            self.params.c_neg
        }
    }

    fn swap_index(&mut self, i: usize, j: usize) {
        self.kernel_matrix.swap(i, j);
        self.class.swap(i, j);
    }

    pub fn solve(&mut self) -> SolverStatus {
        let mut status = SolverStatus::new(
            vec![0.0; self.class.len()],
            vec![0.0; self.class.len()],
            &self.kernel_matrix,
            &self.params
        );

        let mut iter = 0;
        let max_iter = max(10000000, 100*self.class.len());
        let mut counter = min(self.class.len(), 1000)+1;
        while iter < max_iter {
            counter -= 1;
            if counter == 0 {
                counter = min(l, 1000);
                if self.params.shrinking {
                    self.do_shrinking();
                }
            }

            let (i, j, is_optimal) = self.select_working_set(&self.kernel_matrix, &self.classes);
            if is_optimal {
                status.reconstruct_gradient();
                let (i, j, is_optimal) = self.select_working_set(&self.kernel_matrix, &self.classes);
                if is_optimal {
                    break;
                } else {
                    // do shrinking next iteration
                    counter = 1;
                }
            }

            iter ++;

            // update alpha[i] and alpha[j]
            status.update_alpha(i, j);

            // update gradient
            status.update_gradient(i, j);

            // update gradient bar
            status.update_gradient_bar(i, j);
        }

        if iter >= max_iter {
            if status.nactive() < self.class.len() {
                status.reconstruct_gradient();
                status.reset_active();
            }
        }

        let rho = status.calculate_rho();

        // calculate object function
        let mut v = 0.0;
        for i in 0..self.class.len() {
            v += status.alpha[i].0 * (status.gradient_objective[i] + status.p[i]);
        }
        let obj = v / 2.0;

        // put back the solution
        let alpha = (0..self.class.len()).map(|i| status.active_set[i]).collect();
    }
}

*/
