use nalgebra::{Quaternion, UnitQuaternion, Vector3};

pub trait Filter {
    fn process_gyro(&mut self, w: &Vector3<f32>, dt: i64);
    fn process_acc(&mut self, a: &Vector3<f32>, dt: i64);
    fn state(&self) -> UnitQuaternion<f32>;
}

pub struct NaiveIntegrationFilter {
    q: UnitQuaternion<f32>,
}

impl Filter for NaiveIntegrationFilter {
    fn process_gyro(&mut self, w: &Vector3<f32>, dt: i64) {
        let mut alpha = 0.5 * (dt as f32 * 1e-6) * w;
        let norm_alpha = alpha.normalize_mut();
        let (w, x, y, z) = if alpha.sum().is_nan() {
            (1.0, 0.0, 0.0, 0.0)
        } else {
            let c = norm_alpha.cos();
            let s = norm_alpha.sin();
            (c, s * alpha.x, s * alpha.y, s * alpha.z)
        };
        self.q *= UnitQuaternion::from_quaternion(Quaternion::new(w, x, y, z));
    }

    fn process_acc(&mut self, _a: &Vector3<f32>, _dt: i64) {}

    fn state(&self) -> UnitQuaternion<f32> {
        self.q
    }
}

impl NaiveIntegrationFilter {
    pub fn new() -> Self {
        NaiveIntegrationFilter {
            q: UnitQuaternion::identity(),
        }
    }
}
