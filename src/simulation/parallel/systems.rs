use super::{ScanMode, add_active_offsets, add_block_offsets, scan_active_blocks, scan_blocks};
use crate::simulation::{
    pass::GpuPass,
    system::{System, gpu_passes},
};
use cubecl::prelude::*;

#[derive(CubeType, CubeLaunch)]
pub(super) struct BlockScanInput {
    pub buffer: Box<[u32]>,
    pub offset: u32,
    pub count: u32,
    #[cube(comptime)]
    pub mode: ScanMode,
}

pub(super) struct BlockScan;
pub(super) struct ScanBlocks;

impl System for BlockScan {
    type Input = BlockScanInput;
    type Passes = gpu_passes![ScanBlocks];
}

#[cube]
impl GpuPass<BlockScanInput> for ScanBlocks {
    type Output = ();

    fn execute(mut input: BlockScanInput) {
        scan_blocks(
            &mut input.buffer,
            input.offset,
            input.count,
            comptime!(input.mode),
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct BlockOffsetInput {
    pub buffer: Box<[u32]>,
    pub offset: u32,
    pub count: u32,
}

pub(super) struct BlockOffsets;
pub(super) struct AddBlockOffsets;

impl System for BlockOffsets {
    type Input = BlockOffsetInput;
    type Passes = gpu_passes![AddBlockOffsets];
}

#[cube]
impl GpuPass<BlockOffsetInput> for AddBlockOffsets {
    type Output = ();

    fn execute(mut input: BlockOffsetInput) {
        add_block_offsets(&mut input.buffer, input.offset, input.count);
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct ActiveScanInput {
    pub buffer: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub offset: u32,
    pub bound: u32,
    pub divisor: u32,
    #[cube(comptime)]
    pub items: u32,
    #[cube(comptime)]
    pub mode: ScanMode,
}

pub(super) struct ActiveBlockScan;
pub(super) struct ScanActiveBlocks;

impl System for ActiveBlockScan {
    type Input = ActiveScanInput;
    type Passes = gpu_passes![ScanActiveBlocks];
}

#[cube]
impl GpuPass<ActiveScanInput> for ScanActiveBlocks {
    type Output = ();

    fn execute(mut input: ActiveScanInput) {
        scan_active_blocks(
            &mut input.buffer,
            &input.control,
            input.offset,
            input.bound,
            input.divisor,
            comptime!(input.items),
            comptime!(input.mode),
        );
    }
}

#[derive(CubeType, CubeLaunch)]
pub(super) struct ActiveOffsetInput {
    pub buffer: Box<[u32]>,
    pub control: Box<[Atomic<u32>]>,
    pub offset: u32,
    pub bound: u32,
    pub divisor: u32,
    #[cube(comptime)]
    pub items: u32,
}

pub(super) struct ActiveBlockOffsets;
pub(super) struct AddActiveOffsets;

impl System for ActiveBlockOffsets {
    type Input = ActiveOffsetInput;
    type Passes = gpu_passes![AddActiveOffsets];
}

#[cube]
impl GpuPass<ActiveOffsetInput> for AddActiveOffsets {
    type Output = ();

    fn execute(mut input: ActiveOffsetInput) {
        add_active_offsets(
            &mut input.buffer,
            &input.control,
            input.offset,
            input.bound,
            input.divisor,
            comptime!(input.items),
        );
    }
}
