use kiss3d::nalgebra::Vector3;
use rand::{rngs::ThreadRng, thread_rng};
use rand_distr::{Distribution, StandardNormal, Uniform};

pub type AccelerometerNoise = Vector3<f32>;
pub type GyroscopeNoise = Vector3<f32>;

pub struct AccelerometerMeasurement {
    pub dt: i64,
    pub a: Vector3<f32>,
}

pub struct GyroscopeMeasurement {
    pub dt: i64,
    pub w: Vector3<f32>,
}

pub enum ImuMeasurement {
    Accelerometer(AccelerometerMeasurement),
    Gyroscope(GyroscopeMeasurement),
}

pub enum NoisyImuMeasurement {
    Accelerometer(AccelerometerMeasurement, AccelerometerNoise),
    Gyroscope(GyroscopeMeasurement, GyroscopeNoise),
}

pub struct Noisifier {
    dist: StandardNormal,
    rng: ThreadRng,
}

impl Noisifier {
    pub fn new() -> Self {
        let rng = thread_rng();
        let dist = StandardNormal;
        Noisifier { dist, rng }
    }

    pub fn noisify(
        &mut self,
        samples: Vec<ImuMeasurement>,
        acc_noise_stddev: f32,
        gyr_noise_stddev: f32,
    ) -> Vec<NoisyImuMeasurement> {
        samples
            .into_iter()
            .map(|meas| match meas {
                ImuMeasurement::Accelerometer(a) => NoisyImuMeasurement::Accelerometer(
                    a,
                    acc_noise_stddev
                        * Vector3::new(
                            self.dist.sample(&mut self.rng),
                            self.dist.sample(&mut self.rng),
                            self.dist.sample(&mut self.rng),
                        ),
                ),
                ImuMeasurement::Gyroscope(w) => NoisyImuMeasurement::Gyroscope(
                    w,
                    gyr_noise_stddev
                        * Vector3::new(
                            self.dist.sample(&mut self.rng),
                            self.dist.sample(&mut self.rng),
                            self.dist.sample(&mut self.rng),
                        ),
                ),
            })
            .collect()
    }
}

pub struct SimIMU {
    acc_leftover_time: i64,
    acc_sampling_period_us: i64,
    gyr_leftover_time: i64,
    gyr_sampling_period_us: i64,
    dist: Uniform<f32>,
    rng: ThreadRng,
}

impl SimIMU {
    pub fn new(acc_sampling_period: i64, gyr_sampling_period: i64) -> SimIMU {
        let rng = thread_rng();
        let dist = Uniform::new(0.0, std::f32::consts::PI);
        SimIMU {
            acc_leftover_time: 0,
            acc_sampling_period_us: acc_sampling_period,
            gyr_leftover_time: 0,
            gyr_sampling_period_us: gyr_sampling_period,
            rng,
            dist,
        }
    }

    pub fn sample(&mut self, t: i64) -> Vec<ImuMeasurement> {
        let mut meas = Vec::new();

        let mut acc_dt = self.acc_leftover_time;
        while acc_dt + self.acc_sampling_period_us <= t {
            acc_dt += self.acc_sampling_period_us;
            let a = Vector3::new(0.0, 0.0, 9.81);
            meas.push(ImuMeasurement::Accelerometer(AccelerometerMeasurement {
                dt: acc_dt,
                a,
            }));
        }
        self.acc_leftover_time = acc_dt - t;

        let mut gyr_dt = self.gyr_leftover_time;
        while gyr_dt + self.gyr_sampling_period_us <= t {
            gyr_dt += self.gyr_sampling_period_us;
            let theta = self.dist.sample(&mut self.rng);
            let phi = 2.0 * self.dist.sample(&mut self.rng);
            // 2pi/3 = 1 turn in 3 seconds
            let r = 2.0 * std::f32::consts::PI / 1.0;
            let w = r * Vector3::new(
                phi.cos() * theta.sin(),
                phi.sin() * theta.sin(),
                theta.cos(),
            );
            meas.push(ImuMeasurement::Gyroscope(GyroscopeMeasurement {
                dt: gyr_dt,
                w,
            }));
        }
        self.gyr_leftover_time = gyr_dt - t;

        meas.sort_by_key(|m| match m {
            ImuMeasurement::Accelerometer(m) => m.dt,
            ImuMeasurement::Gyroscope(m) => m.dt,
        });

        meas
    }
}
