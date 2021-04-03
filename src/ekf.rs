use log::{debug, warn};
use nalgebra::{
    Cholesky, Matrix3, Matrix3x4, Matrix4, Quaternion, UnitQuaternion, Vector3, Vector4,
};

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
        debug!("EKF predict");
        debug!("Gyroscope values:\t{}", w.transpose());
        debug!("dt:\t\t\t{}", dt);
        let x_vec = Vector4::new(self.x.w, self.x.i, self.x.j, self.x.k);
        let f = Ekf::f(&w, dt);
        debug!("f:\t\t\t{}", f);
        let x_hat = f * x_vec;
        debug!("x_n-1_n-1:\t\t\t{}", x_hat);
        self.x =
            UnitQuaternion::from_quaternion(Quaternion::new(x_hat.x, x_hat.y, x_hat.z, x_hat.w));
        debug!("x_n_n-1:\t\t\t{}", x_hat);
        debug!("p:\t\t\t{}", self.p);
        self.p = f * self.p * f.transpose() + 10.0 * self.q;
        debug!("P_n_n-1:\t\t\t{}", self.p);
        debug!("norm(P_n_n-1):\t\t\t{}", self.p.norm());
    }

    #[allow(clippy::many_single_char_names)]
    fn process_acc(&mut self, a: &Vector3<f32>, _dt: i64) {
        debug!("EKF correct");
        let z = 9.81 * a.normalize();
        debug!("z:\t\t\t{}", z);
        let z_est = self.x.transform_vector(&Vector3::new(0.0, 0.0, 9.81));
        debug!("H(x_n-n-1):\t\t\t{}", z_est);
        let y = z - z_est;
        debug!("y:\t\t\t{}", y);
        let dh_dx = Ekf::dqaqc_dq(&self.x, &Vector3::new(0.0, 0.0, 9.81));
        debug!("dh_dx:\t\t{}", dh_dx);
        let s = dh_dx * self.p * dh_dx.transpose() + self.r;
        debug!("s:\t\t\t{}", s);
        let s_inv = match Cholesky::new(s) {
            Some(c) => c.inverse(),
            None => {
                warn!("s cannot be inverted!?");
                return;
            }
        };
        debug!("s_inv:\t\t\t{}", s_inv);
        let k = self.p * dh_dx.transpose();
        debug!("k:\t\t\t{}", k);
        let correction = k * y;
        debug!("correction:\t\t\t{}", correction);
        let x_prenorm = self.x.quaternion()
            + Quaternion::new(correction.x, correction.y, correction.z, correction.w);
        debug!("x_n_n-1:\t\t\t{}", self.x);
        self.x = UnitQuaternion::from_quaternion(x_prenorm);
        debug!("x_n_n:\t\t\t{}", self.x);
        debug!("P_n_n-1:\t\t\t{}", self.p);
        self.p = (Matrix4::identity() - k * dh_dx) * self.p;
        debug!("P_n_n:\t\t\t{}", self.p);
        debug!("norm(P_n_n):\t\t\t{}", self.p.norm());
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
