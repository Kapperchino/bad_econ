use super::*;
use crate::simulation::{pass::GpuPass, system::gpu_passes};
use cubecl::prelude::*;
use std::marker::PhantomData;

pub(super) struct IterationBuffers {
    pub current: CubeTensor,
    pub next: CubeTensor,
}

impl IterationBuffers {
    pub fn new(device: &Device, initial: &DeviceVector) -> Self {
        let current = tensor::empty(device, [2, initial.length, 1]);
        if initial.length > 0 {
            tensor::copy(
                device,
                &tensor::reshape(&initial.data, [2, initial.length, 1]),
                &current,
            );
        }
        Self {
            current,
            next: tensor::empty(device, [2, initial.length, 1]),
        }
    }

    pub fn step(
        &self,
        device: &Device,
        operator: &DeviceOperator,
        rhs: &DeviceVector,
        control: &CubeTensor,
    ) -> Result<()> {
        tensor::matmul(device, &operator.matrix.data, &self.current, &self.next)?;
        let launch = match tensor::vector_width(device, self.next.meta.num_elements()) {
            4 => update::<Const<4>>,
            3 => update::<Const<3>>,
            2 => update::<Const<2>>,
            _ => update::<Const<1>>,
        };
        launch(
            device,
            &self.next,
            &rhs.data,
            control,
            operator.matrix.scale as f32,
        );
        Ok(())
    }

    pub fn advance(&mut self) {
        std::mem::swap(&mut self.current, &mut self.next);
    }
}

fn update<N: Size>(
    device: &Device,
    values: &CubeTensor,
    rhs: &CubeTensor,
    control: &CubeTensor,
    scale: f32,
) {
    let length = values.meta.num_elements();
    IterationUpdate::<N>::launch(
        device,
        device.count(length / N::value()),
        CubeDim::new_1d(device.workgroup),
        IterationInputLaunch {
            values: argument(values),
            rhs: argument(rhs),
            control: argument(control),
            scale,
            length: length as u32,
        },
    );
}

#[derive(CubeType, CubeLaunch)]
struct IterationInput<N: Size> {
    values: Box<[Vector<f32, N>]>,
    rhs: Box<[Vector<f32, N>]>,
    control: Box<[Atomic<u32>]>,
    scale: f32,
    length: u32,
}

#[derive(CubeType)]
struct RescaledIterate<N: Size> {
    buffers: IterationInput<N>,
    value: Vector<f32, N>,
}

struct IterationUpdate<N: Size>(PhantomData<N>);
struct RescaleIterate;
struct AddForcing;

impl<N: Size> System for IterationUpdate<N> {
    type Input = IterationInput<N>;
    type Passes = gpu_passes![RescaleIterate, AddForcing];
}

#[cube]
impl<N: Size> GpuPass<IterationInput<N>> for RescaleIterate {
    type Output = RescaledIterate<N>;

    fn execute(input: IterationInput<N>) -> RescaledIterate<N> {
        let mut value = Vector::<f32, N>::new(0.0);
        if ABSOLUTE_POS < input.length as usize / N::value() {
            value = kernels::checked_vector(
                input.values[ABSOLUTE_POS] / Vector::new(input.scale),
                &input.control,
            );
        }
        RescaledIterate::<N> {
            buffers: input,
            value,
        }
    }
}

#[cube]
impl<N: Size> GpuPass<RescaledIterate<N>> for AddForcing {
    type Output = ();

    fn execute(mut input: RescaledIterate<N>) {
        if ABSOLUTE_POS < input.buffers.length as usize / N::value() {
            input.buffers.values[ABSOLUTE_POS] = kernels::checked_vector(
                input.value + input.buffers.rhs[ABSOLUTE_POS],
                &input.buffers.control,
            );
        }
    }
}
