use crate::simulation::{
    gpu::device::MAX_WORKGROUP_SIZE,
    storage::{
        flags::KernelFaults,
        words::{Ticks, plus},
    },
};
use cubecl::prelude::*;

#[cube]
pub(in crate::simulation) trait TreeStorage: CubeType {
    fn merge(&mut self, destination: usize, source: usize);
}

pub(in crate::simulation) struct WorkgroupTree;

#[cube]
impl WorkgroupTree {
    pub fn planes<T: CubePrimitive>() -> Shared<[T]> {
        let hardware = comptime::hardware_properties().comptime();
        Shared::<[T]>::new_slice(comptime!(
            MAX_WORKGROUP_SIZE.div_ceil(hardware.plane_size_min.max(1)) as usize
        ))
    }

    pub fn lanes<T: CubePrimitive>(value: T) -> Shared<[T]> {
        let mut lanes = Shared::<[T]>::new_slice(MAX_WORKGROUP_SIZE as usize);
        lanes[UNIT_POS as usize] = value;
        lanes
    }

    pub fn reduce<S: TreeStorage>(storage: &mut S) {
        sync_cube();
        let lane = UNIT_POS as usize;
        #[unroll]
        for level in 0..comptime!(MAX_WORKGROUP_SIZE.ilog2()) {
            let stride = CUBE_DIM >> (level + 1);
            if UNIT_POS < stride {
                storage.merge(lane, lane + stride as usize);
            }
            if stride > 0 {
                sync_cube();
            }
        }
    }

    pub fn inclusive_scan<S: TreeStorage>(storage: &mut S) {
        sync_cube();
        let mut stride = 1u32;
        while stride < CUBE_DIM {
            let end = (UNIT_POS + 1) * stride * 2 - 1;
            if end < CUBE_DIM {
                storage.merge(end as usize, (end - stride) as usize);
            }
            sync_cube();
            stride *= 2;
        }
        stride = CUBE_DIM / 4;
        while stride > 0 {
            let end = (UNIT_POS + 1) * stride * 2 - 1;
            if end + stride < CUBE_DIM {
                storage.merge((end + stride) as usize, end as usize);
            }
            sync_cube();
            stride /= 2;
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(in crate::simulation) enum Reduction {
    Sum,
    Minimum,
    Maximum,
}

#[derive(CubeType)]
struct WorkgroupValues<T: Numeric> {
    lanes: Shared<[T]>,
    #[cube(comptime)]
    operation: Reduction,
}

#[cube]
impl<T: Numeric> WorkgroupValues<T> {
    fn new(value: T, #[comptime] operation: Reduction) -> Self {
        WorkgroupValues::<T> {
            lanes: WorkgroupTree::lanes::<T>(value),
            operation,
        }
    }
}

#[cube]
impl<T: Numeric> TreeStorage for WorkgroupValues<T> {
    fn merge(&mut self, destination: usize, source: usize) {
        let a = self.lanes[destination];
        let b = self.lanes[source];
        self.lanes[destination] = match comptime!(self.operation) {
            Reduction::Sum => a + b,
            Reduction::Minimum => min(a, b),
            Reduction::Maximum => max(a, b),
        };
    }
}

#[cube]
pub(in crate::simulation) fn reduce<T: Numeric>(value: T, #[comptime] operation: Reduction) -> T {
    let mut values = WorkgroupValues::new(value, operation);
    WorkgroupTree::reduce::<WorkgroupValues<T>>(&mut values);
    values.lanes[0]
}

#[derive(CubeType)]
pub(in crate::simulation) struct PrefixSum<T: CubePrimitive = u32> {
    pub inclusive: T,
    pub total: T,
}

#[cube]
pub(in crate::simulation) fn prefix_sum<T: Numeric>(value: T) -> PrefixSum<T> {
    let mut values = WorkgroupValues::new(value, comptime!(Reduction::Sum));
    WorkgroupTree::inclusive_scan::<WorkgroupValues<T>>(&mut values);
    PrefixSum::<T> {
        inclusive: values.lanes[UNIT_POS as usize],
        total: values.lanes[CUBE_DIM as usize - 1],
    }
}

#[derive(CubeType)]
pub(in crate::simulation) struct WideSum {
    pub value: u64,
    pub overflow: u32,
}

#[derive(CubeType)]
pub(in crate::simulation) struct WideSumTree {
    pub values: Shared<[u64]>,
    pub flags: Shared<[u32]>,
}

#[cube]
impl WideSumTree {
    pub fn new(value: u64, flags: u32) -> Self {
        WideSumTree {
            values: WorkgroupTree::lanes::<u64>(value),
            flags: WorkgroupTree::lanes::<u32>(flags),
        }
    }
}

#[cube]
impl TreeStorage for WideSumTree {
    fn merge(&mut self, destination: usize, source: usize) {
        let previous = self.values[destination];
        let sum = previous + self.values[source];
        self.flags[destination] |= self.flags[source] | u32::cast_from(sum < previous);
        self.values[destination] = sum;
    }
}

#[cube]
pub(in crate::simulation) fn sum_wide(value: u64) -> WideSum {
    let mut tree = WideSumTree::new(value, 0);
    WorkgroupTree::reduce::<WideSumTree>(&mut tree);
    let result = WideSum {
        value: tree.values[0],
        overflow: tree.flags[0],
    };
    sync_cube();
    result
}

#[derive(CubeType)]
pub(in crate::simulation) struct TickTree {
    pub values: Shared<[Ticks]>,
    pub faults: KernelFaults,
}

#[cube]
impl TreeStorage for TickTree {
    fn merge(&mut self, destination: usize, source: usize) {
        let sum = plus(self.values[destination], self.values[source]);
        self.values[destination] = sum.value;
        self.faults.insert(sum.faults);
    }
}

#[cube]
impl TickTree {
    pub fn new(value: Ticks) -> Self {
        TickTree {
            values: WorkgroupTree::lanes::<Ticks>(value),
            faults: KernelFaults::none(),
        }
    }
}
