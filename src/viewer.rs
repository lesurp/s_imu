use std::collections::HashMap;

use crate::filter::Filter;
use kiss3d::{
    camera::FirstPerson,
    nalgebra::{Point3, Translation3, UnitQuaternion, Vector3},
    scene::SceneNode,
    window::Window,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct FilterNodeHandler(usize);

pub struct Viewer {
    w: Window,
    camera: FirstPerson,
    filter_nodes: HashMap<FilterNodeHandler, SceneNode>,
    scale: f32,
}

impl Viewer {
    pub const CYLINDER_RADIUS: f32 = 0.1;
    pub const CYLINDER_LENGTH: f32 = 10.0;
    pub fn new(name: &str, scale: f32) -> Viewer {
        let w = Window::new(name);

        let mut camera = FirstPerson::new(Point3::new(1.0, 4.0, 2.0), Point3::new(0.0, 0.0, 0.0));
        camera.set_up_axis_dir(Vector3::z_axis());

        Viewer {
            w,
            camera,
            filter_nodes: HashMap::new(),
            scale,
        }
    }

    pub fn update<F: Filter>(&mut self, h: FilterNodeHandler, f: &F) {
        self.filter_nodes
            .get_mut(&h)
            .unwrap()
            .set_local_rotation(f.state());
    }

    pub fn add_filter(
        &mut self,
        offset: Vector3<f32>,
        r: f32,
        g: f32,
        b: f32,
    ) -> FilterNodeHandler {
        let mut xyz_gt = Viewer::draw_coordinate_system(&mut self.w, self.scale);
        let mut gt_distinguish = xyz_gt.add_sphere(self.scale * 0.3);
        xyz_gt.set_local_translation(Translation3::from(self.scale * offset));
        gt_distinguish.set_color(r, g, b);
        let id = FilterNodeHandler(self.filter_nodes.len());
        self.filter_nodes.insert(id, xyz_gt);
        id
    }

    fn draw_coordinate_system(w: &mut Window, scale: f32) -> SceneNode {
        let h = scale * Viewer::CYLINDER_LENGTH;
        let r = scale * Viewer::CYLINDER_RADIUS;

        let mut xyz = w.add_group();
        let mut x_axis = xyz.add_cylinder(r, h);
        let mut y_axis = xyz.add_cylinder(r, h);
        let mut z_axis = xyz.add_cylinder(r, h);

        x_axis.set_local_translation(Translation3::new(h / 2.0, 0.0, 0.0));
        y_axis.set_local_translation(Translation3::new(0.0, h / 2.0, 0.0));
        z_axis.set_local_translation(Translation3::new(0.0, 0.0, h / 2.0));

        x_axis.set_color(1.0, 0.0, 0.0);
        y_axis.set_color(0.0, 1.0, 0.0);
        z_axis.set_color(0.0, 0.0, 1.0);

        x_axis.set_local_rotation(UnitQuaternion::from_axis_angle(
            &Vector3::z_axis(),
            -std::f32::consts::FRAC_PI_2,
        ));
        z_axis.set_local_rotation(UnitQuaternion::from_axis_angle(
            &Vector3::x_axis(),
            std::f32::consts::FRAC_PI_2,
        ));

        xyz
    }

    pub fn render(&mut self) -> bool {
        self.w.render_with_camera(&mut self.camera)
    }
}
