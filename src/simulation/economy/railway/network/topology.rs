use super::{Device, ROUTE_WORDS, SEGMENT_WORDS, UNREACHABLE};
use crate::domain::{
    Position,
    railway::{RailGeography, RailwayConfig, RouteStorage},
};
use burn_wgpu::CubeTensor;
use std::collections::BTreeSet;

pub(super) struct NetworkTopology {
    pub(super) segments: Vec<u32>,
    pub(super) segment_regions: Vec<u32>,
    pub(super) regions: Vec<u32>,
    station_companies: Vec<BTreeSet<u32>>,
    incident_segments: Vec<Vec<u32>>,
}

pub(super) struct RouteTables {
    pub(super) index: usize,
    pub(super) rows: usize,
    pub(super) offsets: Vec<u32>,
    pub(super) routes: CubeTensor,
}

impl NetworkTopology {
    pub(super) fn new(
        config: &RailwayConfig,
        geography: &RailGeography,
        stations: &[Position],
    ) -> Self {
        let mut segment_regions = Vec::new();
        let mut station_companies = vec![BTreeSet::new(); stations.len()];
        let mut incident_segments = vec![Vec::new(); stations.len()];
        let mut ordinals = vec![0; config.companies.len()];
        let mut first_segments = vec![u32::MAX; geography.regions.len()];
        let segments = config
            .segments
            .iter()
            .enumerate()
            .flat_map(|(index, segment)| {
                let company = config
                    .companies
                    .iter()
                    .position(|company| company.firm == segment.company)
                    .expect("railway company");
                let ordinal = ordinals[company];
                ordinals[company] += 1;
                for &region in &geography.segments[index] {
                    first_segments[region] = first_segments[region].min(ordinal);
                }
                let offset = segment_regions.len();
                segment_regions.extend(
                    geography.segments[index]
                        .iter()
                        .map(|region| *region as u32),
                );
                let from = stations
                    .iter()
                    .position(|position| *position == segment.from)
                    .expect("station");
                let to = stations
                    .iter()
                    .position(|position| *position == segment.to)
                    .expect("station");
                station_companies[from].insert(company as u32);
                station_companies[to].insert(company as u32);
                incident_segments[from].push(index as u32);
                incident_segments[to].push(index as u32);
                [
                    from as u32,
                    to as u32,
                    company as u32,
                    segment.from.distance(segment.to),
                    ordinal,
                    offset as u32,
                    geography.segments[index].len() as u32,
                ]
            })
            .collect::<Vec<_>>();
        let regions = geography
            .regions
            .iter()
            .enumerate()
            .flat_map(|(index, region)| {
                [
                    region.company as u32,
                    region.region as u32,
                    first_segments[index],
                ]
            })
            .collect::<Vec<_>>();
        Self {
            segments,
            segment_regions,
            regions,
            station_companies,
            incident_segments,
        }
    }

    pub(super) fn route_tables(
        &self,
        device: &Device,
        companies: usize,
        storage: RouteStorage,
    ) -> RouteTables {
        let stations = self.station_companies.len();
        let carriers = self
            .station_companies
            .iter()
            .map(|companies| companies.len())
            .sum::<usize>();
        let mut offsets = Vec::with_capacity(
            2 * (stations + 1) + carriers + 2 * (self.segments.len() / SEGMENT_WORDS),
        );
        offsets.push(stations as u32 + 1);
        offsets.extend(
            self.station_companies
                .iter()
                .scan(stations + 1, |offset, companies| {
                    *offset += companies.len();
                    Some(*offset as u32)
                }),
        );
        offsets.extend(self.station_companies.iter().flatten().copied());
        let incident_start = offsets.len() + stations + 1;
        offsets.push(incident_start as u32);
        offsets.extend(
            self.incident_segments
                .iter()
                .scan(incident_start, |offset, segments| {
                    *offset += segments.len();
                    Some(*offset as u32)
                }),
        );
        offsets.extend(self.incident_segments.iter().flatten().copied());
        let dense = stations * companies;
        match storage {
            RouteStorage::Indexed => {
                let rows = carriers;
                let index = offsets.len();
                let stations_count = stations as u32;
                let origins = self
                    .station_companies
                    .iter()
                    .enumerate()
                    .flat_map(|(origin, owners)| {
                        owners
                            .iter()
                            .map(move |&company| company * stations_count + origin as u32)
                    })
                    .collect::<Vec<_>>();
                let mut lookup = vec![rows as u32; dense];
                for (row, &pair) in origins.iter().enumerate() {
                    lookup[pair as usize] = row as u32;
                }
                offsets.extend(lookup);
                offsets.extend(origins);
                let empty = [UNREACHABLE, u32::MAX, u32::MAX, 0, 0, UNREACHABLE];
                RouteTables {
                    index,
                    rows,
                    offsets,
                    routes: device.tensor(&empty.repeat((rows + 1) * stations)),
                }
            }
            RouteStorage::Dense => RouteTables {
                index: 0,
                rows: dense,
                offsets,
                routes: device.empty((dense * stations * ROUTE_WORDS).max(1)),
            },
        }
    }
}
