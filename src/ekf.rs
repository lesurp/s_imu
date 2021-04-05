use log::{debug, warn};
use nalgebra::{Matrix3, Matrix3x4, Matrix4, Quaternion, UnitQuaternion, Vector3, Vector4, QR};

use crate::filter::Filter;

pub struct Ekf {
    x: UnitQuaternion<f32>,
    p: Matrix4<f32>,
    q: Matrix4<f32>,
    r: Matrix3<f32>,
}

impl Ekf {
    pub fn new(x: UnitQuaternion<f32>, p: Matrix4<f32>, acc_cov: f32, gyr_cov: f32) -> Ekf {
        let r = acc_cov * Matrix3::identity();
        let q = Matrix4::from_diagonal(&Vector4::new(gyr_cov, gyr_cov, gyr_cov, gyr_cov));
        Ekf { x, p, q, r }
    }
}

impl Filter for Ekf {
    fn process_gyro(&mut self, w: &Vector3<f32>, dt: i64) {
        let x_vec = Vector4::new(self.x.w, self.x.i, self.x.j, self.x.k);
        let f = Ekf::f(&w, dt);
        let x_hat = f * x_vec;

        debug!("EKF predict");
        debug!("Gyroscope values:\t{}", w.transpose());
        debug!("dt:\t\t\t{}", dt);
        debug!("f:\t\t\t{}", f);
        debug!("x_n-1_n-1:\t\t\t{}", x_hat);
        debug!("P_n-1_n-1:\t\t\t{}", self.p);

        self.x =
            UnitQuaternion::from_quaternion(Quaternion::new(x_hat.x, x_hat.y, x_hat.z, x_hat.w));
        self.p = f * self.p * f.transpose() + self.q;

        debug!("x_n_n-1:\t\t\t{}", x_hat);
        debug!("P_n_n-1:\t\t\t{}", self.p);
    }

    #[allow(clippy::many_single_char_names)]
    fn process_acc(&mut self, a: &Vector3<f32>, _dt: i64) {
        let z = 9.81 * a.normalize(); // TODO: idiotic way of getting gravity from acc: this should be another module's responsability!
        let hx = self.x * Vector3::new(0.0, 0.0, 9.81);
        let y = z - hx;
        let dh_dx = Ekf::dqaqc_dq(&self.x, &Vector3::new(0.0, 0.0, 9.81));

        let s = dh_dx * self.p * dh_dx.transpose() + self.r;

        debug!("EKF correct");
        debug!("x_n-n-1:\t\t\t{}", self.x);
        debug!("z:\t\t\t{}", z);
        debug!("H(x_n-n-1):\t\t\t{}", hx);
        debug!("y:\t\t\t{}", y);
        debug!("dh_dx:\t\t{}", dh_dx);
        debug!("s:\t\t\t{}", s);
        let s_inv = match QR::new(s).try_inverse() {
            Some(c) => {
                if c.sum().is_nan() {
                    warn!("s.inv has NaN inverted!?");
                    return;
                } else {
                    c
                }
            }
            None => {
                warn!("s cannot be inverted!?");
                return;
            }
        };

        let k = self.p * dh_dx.transpose() * s_inv;
        let correction = k * y;
        // we compute p (because of f) and dh_dx with w first
        // we do NOT usethe .x .y etc accessors to avoid needless confusion: correction[0] is our
        // quaternion's real part
        let x_prenorm = self.x.quaternion()
            + Quaternion::new(correction[0], correction[1], correction[2], correction[3]);


        debug!("s_inv:\t\t\t{}", s_inv);
        debug!("k:\t\t\t{}", k);
        debug!("correction:\t\t\t{}", correction);
        debug!("x_n_n-1:\t\t\t{}", self.x);
        debug!("P_n_n-1:\t\t\t{}", self.p);

        self.x = UnitQuaternion::from_quaternion(x_prenorm);
        self.p = (Matrix4::identity() - k * dh_dx) * self.p;

        debug!("x_n_n:\t\t\t{}", self.x);
        debug!("P_n_n:\t\t\t{}", self.p);
    }

    fn state(&self) -> UnitQuaternion<f32> {
        self.x
    }
}

impl Ekf {
    fn dqaqc_dq(x: &UnitQuaternion<f32>, a: &Vector3<f32>) -> Matrix3x4<f32> {
        let a_skewed = Ekf::skewed(&a);
        let w = x.w;
        let v = x.vector();
        let dqgqc_dw = w * a + v.cross(&a);
        let dqgqc_dv =
            v.dot(&a) * Matrix3::identity() + v * a.transpose() - a * v.transpose() - w * a_skewed;

        let mut out = Matrix3x4::zeros();
        out.set_column(0, &dqgqc_dw);
        for i in 0..3 {
            out.set_column(i + 1, &dqgqc_dv.column(i));
        }

        2.0 * out
    }

    pub fn skewed(a: &Vector3<f32>) -> Matrix3<f32> {
        let mut a_skewed = Matrix3::zeros();
        a_skewed[(0, 1)] = -a.z;
        a_skewed[(0, 2)] = a.y;
        a_skewed[(1, 0)] = a.z;
        a_skewed[(1, 2)] = -a.x;
        a_skewed[(2, 0)] = -a.y;
        a_skewed[(2, 1)] = a.x;
        a_skewed
    }

    pub fn p(&self) -> Matrix4<f32> {
        self.p
    }

    #[allow(clippy::many_single_char_names)]
    fn f(w: &Vector3<f32>, dt: i64) -> Matrix4<f32> {
        let exp_w = 0.5 * (dt as f32 * 1e-6) * w;
        let mut f = Matrix4::identity();
        let x = exp_w.x;
        let y = exp_w.y;
        let z = exp_w.z;

        f[(0, 1)] = -x;
        f[(0, 2)] = -y;
        f[(0, 3)] = -z;

        f[(1, 0)] = x;
        f[(1, 2)] = z;
        f[(1, 3)] = -y;

        f[(2, 0)] = y;
        f[(2, 1)] = -z;
        f[(2, 3)] = x;

        f[(3, 0)] = z;
        f[(3, 1)] = y;
        f[(3, 2)] = -x;

        f
    }
}

#[cfg(test)]
mod test {
    use nalgebra::{Matrix4, UnitQuaternion, Vector3};

    use crate::filter::Filter;

    use super::Ekf;
    fn ekf() -> Ekf {
        Ekf::new(UnitQuaternion::identity(), Matrix4::zeros(), 1e-1, 1e-1)
    }

    #[test]
    fn test_p_increase() {
        let mut ekf = ekf();
        let mut prev_p = ekf.p();
        for _ in 0..10 {
            ekf.process_gyro(&Vector3::zeros(), 1);
            let p = ekf.p();
            assert!(prev_p.norm() < p.norm());
            prev_p = p;
        }
    }

    #[test]
    fn test_p_decrease() {
        let mut ekf = ekf();
        for _ in 0..10 {
            ekf.process_gyro(&Vector3::zeros(), 1);
            let prev_p = ekf.p();
            ekf.process_acc(&Vector3::x(), 1);
            let p = ekf.p();
            assert!(prev_p.norm() > p.norm());
        }
    }

    #[test]
    fn test_predict() {
        let mut ekf = ekf();
        ekf.x = UnitQuaternion::new(Vector3::new(0.0, std::f32::consts::FRAC_PI_2, 0.0));

        // increase p artificially
        for _ in 0..10 {
            ekf.process_gyro(&Vector3::zeros(), 1);
        }

        let g = Vector3::new(0.0, 0.0, 9.81);
        for _ in 0..100 {
            ekf.process_acc(&g, 1);
        }

        let err = ekf.x.angle_to(&UnitQuaternion::identity());
        assert!(err < 1e-5);
    }

    #[test]
    fn test_correct() {
        let mut ekf = ekf();

        for _ in 0..100 {
            ekf.process_gyro(&Vector3::new(0.0, std::f32::consts::FRAC_PI_2, 0.0), 10_000);
        }

        let err = ekf.x.angle_to(&UnitQuaternion::new(Vector3::new(
            0.0,
            std::f32::consts::FRAC_PI_2,
            0.0,
        )));
        assert!(err < 1e-4);
    }
}
