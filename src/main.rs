use ekf::Ekf;
use filter::{Filter, NaiveIntegrationFilter};
use kiss3d::nalgebra::Vector3;
use log::info;
use nalgebra::{Matrix4, UnitQuaternion};
use s_imu::{Noisifier, NoisyImuMeasurement, SimIMU};
use std::time::Instant;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use viewer::Viewer;

mod ekf;
mod filter;
mod s_imu;
mod viewer;

fn main() {
    env_logger::init();
    let mut v = Viewer::new("sIMU", 0.1);

    let mut stdout = StandardStream::stdout(ColorChoice::Always);

    let acc_sampling_period_us = 5000;
    let gyr_sampling_period_us = 10000;
    let imu_manufacturer_lying_factor = 10.0;

    let gyro_noise_density = 0.03;
    let gyr_noise_stddev = imu_manufacturer_lying_factor
        * gyro_noise_density
        * (1e6 / gyr_sampling_period_us as f32).sqrt()
        * std::f32::consts::PI
        / 180.0;
    let gyr_cov = gyr_noise_stddev * gyr_noise_stddev;

    let acc_noise_density = 9.81 * 218e-6;
    let acc_noise_stddev = acc_noise_density
        * imu_manufacturer_lying_factor
        * (1e6 / acc_sampling_period_us as f32).sqrt();
    let acc_cov = acc_noise_stddev * acc_noise_stddev;

    info!(
        "Accelerometer noise standard deviation: {}",
        acc_noise_stddev
    );
    info!("Gyroscope noise standard deviation: {}", gyr_noise_stddev);

    let mut gt_filter = NaiveIntegrationFilter::new();
    let mut naive_filter = NaiveIntegrationFilter::new();
    let mut ekf = Ekf::new(
        UnitQuaternion::identity(),
        Matrix4::zeros(),
        acc_cov,
        gyr_cov,
    );

    let gt_filter_handle = v.add_filter(
        Vector3::new(0.0, -1.5 * Viewer::CYLINDER_LENGTH, 0.0),
        0.6,
        0.2,
        0.0,
    );
    let naive_filter_handle = v.add_filter(
        Vector3::new(0.0, 1.5 * Viewer::CYLINDER_LENGTH, 0.0),
        0.2,
        0.6,
        0.6,
    );
    let ekf_handle = v.add_filter(
        Vector3::new(1.5 * Viewer::CYLINDER_LENGTH, 0.0, 0.0),
        1.0,
        0.0,
        0.0,
    );
    let _orig = v.add_filter(Vector3::zeros(), 1.0, 1.0, 1.0);

    let mut simu = SimIMU::new(acc_sampling_period_us, gyr_sampling_period_us);
    let mut noisifier = Noisifier::new();
    let mut time = Instant::now();
    let alpha = 0.50;
    let mut running_avg_naive = 0.0;
    let mut running_avg_ekf = 0.0;
    let mut update_counter = 0;
    loop {
        if !v.render() {
            break;
        }
        let now = Instant::now();
        let s = now - time;
        let elapsed = s.as_micros() as i64;
        time = now;
        let meas = noisifier.noisify(simu.sample(elapsed), acc_noise_stddev, gyr_noise_stddev);
        for m in meas {
            match m {
                NoisyImuMeasurement::Accelerometer(a, an) => {
                    let a_perfect = gt_filter.state().transform_vector(&a.a);
                    let a_noisy = a_perfect + an;
                    ekf.process_acc(&a_noisy, a.dt);
                }
                NoisyImuMeasurement::Gyroscope(g, wn) => {
                    let g_noisy = g.w + wn;
                    gt_filter.process_gyro(&g.w, g.dt);
                    naive_filter.process_gyro(&g_noisy, g.dt);
                    ekf.process_gyro(&g_noisy, g.dt);

                    running_avg_naive = (1.0 - alpha) * running_avg_naive
                        + alpha
                            * (gt_filter.state().as_vector() - naive_filter.state().as_vector())
                                .norm();

                    running_avg_ekf = (1.0 - alpha) * running_avg_ekf
                        + alpha * (gt_filter.state().as_vector() - ekf.state().as_vector()).norm();
                }
            }
        }
        v.update(gt_filter_handle, &gt_filter);
        v.update(naive_filter_handle, &naive_filter);
        v.update(ekf_handle, &ekf);

        update_counter += elapsed;
        if update_counter >= 1_000_000 {
            update_counter -= 1_000_000;
            println!("______");
            println!("Errors:");
            println!("Naive integration:\t{}", running_avg_naive);
            println!("EKF:\t\t\t{}", running_avg_ekf);
            println!("EKF covariance:\t{}", ekf.p());
            println!("EKF covariance norm:\t{}", ekf.p().norm());
            let better = running_avg_ekf <= running_avg_naive;
            if better {
                stdout
                    .set_color(ColorSpec::new().set_fg(Some(Color::Green)))
                    .unwrap();
                println!("EKF is better 😇");
            } else {
                stdout
                    .set_color(ColorSpec::new().set_fg(Some(Color::Red)))
                    .unwrap();
                print!("EKF is worse 😠");
            }
            stdout.reset().unwrap();
            println!();
        }
    }
}
