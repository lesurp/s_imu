use nalgebra::UnitQuaternion;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Q<T, F>(
    pub UnitQuaternion<f32>,
    std::marker::PhantomData<F>,
    std::marker::PhantomData<T>,
);

impl<T, F> Q<T, F> {
    pub fn from(q: UnitQuaternion<f32>) -> Self {
        Q::<T, F>(q, std::marker::PhantomData, std::marker::PhantomData)
    }

    pub fn new() -> Self {
        Q::<T, F>(
            UnitQuaternion::identity(),
            std::marker::PhantomData,
            std::marker::PhantomData,
        )
    }
}

impl<A, B, C> std::ops::Mul<Q<B, C>> for Q<A, B> {
    type Output = Q<A, C>;
    fn mul(self, rhs: Q<B, C>) -> Self::Output {
        Q::from(self.0 * rhs.0)
    }
}

impl<A, B> std::ops::MulAssign<Q<B, B>> for Q<A, B> {
    fn mul_assign(&mut self, rhs: Q<B, B>) {
        self.0 *= rhs.0
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct World;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Local;
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Perturbation;
