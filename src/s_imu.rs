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
    acc_sampling_period: i64,
    gyr_leftover_time: i64,
    gyr_sampling_period: i64,
    dist: Uniform<f32>,
    rng: ThreadRng,
}

impl SimIMU {
    pub fn new(acc_sampling_period: i64, gyr_sampling_period: i64) -> SimIMU {
        let rng = thread_rng();
        let dist = Uniform::new(0.0, std::f32::consts::PI);
        SimIMU {
            acc_leftover_time: 0,
            acc_sampling_period,
            gyr_leftover_time: 0,
            gyr_sampling_period,
            rng,
            dist,
        }
    }

    pub fn sample(&mut self, t: i64) -> Vec<ImuMeasurement> {
        let mut meas = Vec::new();
        {
            let mut dt = self.acc_leftover_time;
            while dt + self.acc_sampling_period <= t {
                dt += self.acc_sampling_period;
                let a = Vector3::new(0.0, 0.0, 9.81);
                meas.push(ImuMeasurement::Accelerometer(AccelerometerMeasurement {
                    dt: self.acc_sampling_period,
                    a,
                }));
            }
            self.acc_leftover_time = dt - t;
        }

        {
            let mut dt = self.gyr_leftover_time;
            while dt + self.gyr_sampling_period <= t {
                dt += self.gyr_sampling_period;
                let theta = 0.0_f32;
                //let theta = self.dist.sample(&mut self.rng);
                let phi = 2.0 * self.dist.sample(&mut self.rng);
                // 2pi/3 = 1 turn in 3 seconds
                let r = 2.0 * std::f32::consts::PI / 3.0;
                let w = r * Vector3::new(
                    phi.cos() * theta.sin(),
                    phi.sin() * theta.sin(),
                    theta.cos(),
                );
                meas.push(ImuMeasurement::Gyroscope(GyroscopeMeasurement {
                    dt: self.gyr_sampling_period,
                    w,
                }));
            }
            self.gyr_leftover_time = dt - t;
        }

        /* TODO
        meas.sort_by_key(|m| match m {
            ImuMeasurement::Gyroscope(g) => g.dt,
            ImuMeasurement::Accelerometer(a) => a.dt,
        });
        */

        meas
    }
}
