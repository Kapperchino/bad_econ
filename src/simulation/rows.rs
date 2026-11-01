macro_rules! row_layout {
    ($rows:ident, $record:ident { $first:ident : $first_ty:ty => $first_constant:ident $(, $field:ident : $ty:ty => $constant:ident)* $(,)? }) => {
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        pub(crate) struct $record {
            pub $first: $first_ty,
            $(pub $field: $ty),*
        }
        impl $record {
            pub const WORDS: usize = size_of::<Self>() / size_of::<u32>();
            pub const $first_constant: usize = std::mem::offset_of!(Self, $first) / size_of::<u32>();
            $(pub const $constant: usize = std::mem::offset_of!(Self, $field) / size_of::<u32>();)*
        }
        #[derive(Default)]
        pub(crate) struct $rows {
            $first: Vec<$first_ty>,
            $($field: Vec<$ty>),*
        }
        impl $rows {
            pub fn len(&self) -> usize {
                self.$first.len()
            }
            pub fn push(&mut self, record: $record) {
                self.$first.push(record.$first);
                $(self.$field.push(record.$field);)*
            }
            pub fn words(&self) -> Vec<u32> {
                let mut words = Vec::with_capacity(self.len() * $record::WORDS);
                words.extend_from_slice(bytemuck::cast_slice(&self.$first));
                $(words.extend_from_slice(bytemuck::cast_slice(&self.$field));)*
                words
            }
        }
        impl FromIterator<$record> for $rows {
            fn from_iter<T: IntoIterator<Item = $record>>(records: T) -> Self {
                records.into_iter().fold(Self::default(), |mut rows, record| {
                    rows.push(record);
                    rows
                })
            }
        }
    };
}
pub(crate) use row_layout;

use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct RowIndex {
    offset: usize,
    count: usize,
    index: usize,
}

#[cube]
impl RowIndex {
    pub fn new(offset: usize, count: usize, index: usize) -> RowIndex {
        RowIndex {
            offset,
            count,
            index,
        }
    }
    pub fn field(&self, #[comptime] field: usize, #[comptime] words: usize) -> usize {
        self.offset + field * self.count + self.index * words
    }
}
