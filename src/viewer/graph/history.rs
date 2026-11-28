use super::{Chart, ChartLine, GraphLine, LineRun};
use crate::{domain::ObservationSnapshot, viewer::model::SeriesSelection};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    ops::{Range, RangeInclusive},
};

#[derive(Clone, Copy)]
pub(super) struct Point {
    pub day: u64,
    pub value: f64,
}

#[derive(Default)]
pub(super) struct Trace {
    pub points: Vec<Point>,
}

impl Trace {
    fn push(&mut self, point: Point) {
        match self.points.as_mut_slice() {
            [.., last] if last.day == point.day => (),
            [.., last] if !last.value.is_finite() && !point.value.is_finite() => (),
            [.., prior, last] if prior.value == last.value && last.value == point.value => {
                *last = point;
            }
            _ => self.points.push(point),
        }
    }

    fn reduce(&mut self, bucket_days: u64) {
        let raw = std::mem::take(&mut self.points);
        for bucket in raw.chunk_by(|a, b| {
            a.day / bucket_days == b.day / bucket_days && a.value.is_finite() == b.value.is_finite()
        }) {
            let first = bucket[0];
            if first.value.is_finite() && bucket.len() > 4 {
                let (mut low, mut high) = (first, first);
                for &point in &bucket[1..] {
                    if point.value < low.value {
                        low = point;
                    }
                    if point.value > high.value {
                        high = point;
                    }
                }
                let mut extrema = [low, high];
                extrema.sort_unstable_by_key(|point| point.day);
                for point in [first, extrema[0], extrema[1], bucket[bucket.len() - 1]] {
                    self.push(point);
                }
            } else {
                for &point in bucket {
                    self.push(point);
                }
            }
        }
    }
}

#[derive(Default)]
struct Block {
    indices: Range<usize>,
    series: Vec<Trace>,
}

impl Block {
    fn new(
        chart: Chart,
        history: &VecDeque<ObservationSnapshot>,
        indices: Range<usize>,
        bucket_days: u64,
    ) -> Self {
        let mut series = Vec::new();
        for sample in indices.clone().map(|index| chart.sample(history, index)) {
            series.resize_with(sample.values.len(), Trace::default);
            for (trace, value) in series.iter_mut().zip(sample.values) {
                trace.points.push(Point {
                    day: sample.day,
                    value,
                });
            }
        }
        for trace in &mut series {
            trace.reduce(bucket_days);
        }
        Self { indices, series }
    }
}

#[derive(Default)]
struct CachedHistory {
    bucket_days: u64,
    blocks: BTreeMap<usize, Block>,
}

#[derive(Default)]
pub(in crate::viewer) struct GraphCache {
    generation: u64,
    histories: HashMap<(Chart, LineRun), CachedHistory>,
}

impl GraphCache {
    pub fn retain(&mut self, generation: u64, charts: impl Iterator<Item = Chart> + Clone) {
        if self.generation != generation {
            self.histories.clear();
            self.generation = generation;
        }
        self.histories
            .retain(|(chart, _), _| charts.clone().any(|active| *chart == active));
    }

    pub(super) fn lines(
        &mut self,
        chart: Chart,
        run: LineRun,
        history: &VecDeque<ObservationSnapshot>,
        days: &RangeInclusive<u64>,
        bucket_days: u64,
        selection: SeriesSelection,
    ) -> Vec<GraphLine> {
        let cache = self.histories.entry((chart, run)).or_default();
        if cache.bucket_days != bucket_days {
            cache.blocks.clear();
            cache.bucket_days = bucket_days;
        }
        let start = history.partition_point(|sample| sample.day < *days.start());
        let end = history.partition_point(|sample| sample.day <= *days.end());
        let block_size = (bucket_days as usize).saturating_mul(8).max(32);
        cache
            .blocks
            .retain(|_, block| start <= block.indices.start && block.indices.end <= end);
        let mut cursor = start;
        while cursor < end {
            let index = cursor / block_size;
            let next = (index * block_size).saturating_add(block_size).min(end);
            let indices = cursor..next;
            let block = cache.blocks.entry(index).or_default();
            if block.indices != indices {
                *block = Block::new(chart, history, indices, bucket_days);
            }
            cursor = next;
        }

        let count = cache
            .blocks
            .values()
            .next()
            .map_or(0, |block| block.series.len());
        (0..count)
            .filter(|&series| selection.includes(series))
            .map(|series| {
                let mut trace = Trace::default();
                for block in cache.blocks.values() {
                    for &point in &block.series[series].points {
                        trace.push(point);
                    }
                }
                GraphLine {
                    id: ChartLine { chart, run, series },
                    trace,
                }
            })
            .collect()
    }
}
