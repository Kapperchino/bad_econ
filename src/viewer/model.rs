use crate::{domain::ObservationSnapshot, runtime::RuntimeState};
use std::collections::VecDeque;

pub enum DisplayState {
    Initializing,
    Active(RuntimeState),
    Unavailable(String),
}
impl DisplayState {
    pub fn label(&self) -> String {
        match self {
            Self::Initializing => "Initializing shared GPU...".into(),
            Self::Active(state) => format!("{state:?}"),
            Self::Unavailable(message) => format!("Fault: {message}"),
        }
    }
    pub fn failed(&self) -> bool {
        matches!(
            self,
            Self::Unavailable(_) | Self::Active(RuntimeState::Fault { .. })
        )
    }
}

pub struct ComparisonRun {
    pub seed: u64,
    pub horizon: u64,
    pub history: VecDeque<ObservationSnapshot>,
}
impl ComparisonRun {
    pub fn at_day(&self, day: u64) -> Option<&ObservationSnapshot> {
        self.history.iter().find(|snapshot| snapshot.day == day)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunTarget {
    Open,
    UntilDay(u64),
}
impl RunTarget {
    pub fn reached(self, day: u64) -> bool {
        match self {
            Self::Open => false,
            Self::UntilDay(horizon) => day >= horizon,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chart {
    Employment,
    Prices,
    Wages,
    Inventories,
    Cash,
}
impl Chart {
    pub fn file_label(self) -> &'static str {
        match self {
            Self::Employment => "employment",
            Self::Prices => "prices",
            Self::Wages => "wages",
            Self::Inventories => "inventories",
            Self::Cash => "cash",
        }
    }
    pub fn next(self) -> Self {
        match self {
            Self::Employment => Self::Prices,
            Self::Prices => Self::Wages,
            Self::Wages => Self::Inventories,
            Self::Inventories => Self::Cash,
            Self::Cash => Self::Employment,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Employment => "Employment (green) / unmet bread (red)",
            Self::Prices => "Mean prices | all goods",
            Self::Wages => "Mean wages | all industries",
            Self::Inventories => "Available stocks | all goods",
            Self::Cash => "Household cash | bins 0, 1, 2-3, 4-7, ...",
        }
    }
    pub fn values(self, snapshot: &ObservationSnapshot) -> Vec<f64> {
        match self {
            Self::Employment => vec![snapshot.employed as f64, snapshot.shortages as f64],
            Self::Prices => snapshot
                .markets
                .iter()
                .map(|market| market.prices.mean())
                .collect(),
            Self::Wages => snapshot
                .markets
                .iter()
                .map(|market| market.wages.mean())
                .collect(),
            Self::Inventories => snapshot
                .inventories
                .iter()
                .map(|value| *value as f64)
                .collect(),
            Self::Cash => snapshot
                .wealth_bins
                .iter()
                .map(|value| *value as f64)
                .collect(),
        }
    }
}

pub struct ChartRange {
    pub first_day: u64,
    pub last_day: u64,
    pub maximum: f64,
    pub last_bin: usize,
}
impl ChartRange {
    pub fn new(
        chart: Chart,
        snapshot: &ObservationSnapshot,
        history: &VecDeque<ObservationSnapshot>,
        comparison: Option<&ComparisonRun>,
    ) -> Self {
        let first_day = history
            .front()
            .map(|point| point.day)
            .unwrap_or(snapshot.day);
        let last_day = snapshot.day;
        let visible = |point: &&ObservationSnapshot| (first_day..=last_day).contains(&point.day);
        let current = match chart {
            Chart::Cash => vec![snapshot],
            _ => history.iter().filter(visible).collect(),
        };
        let reference = match (chart, comparison) {
            (Chart::Cash, Some(reference)) => reference.at_day(last_day).into_iter().collect(),
            (_, Some(reference)) => reference.history.iter().filter(visible).collect(),
            (_, None) => Vec::new(),
        };
        let maximum = current
            .iter()
            .chain(&reference)
            .flat_map(|point| chart.values(point))
            .fold(1.0, f64::max);
        let last_bin = current
            .iter()
            .chain(&reference)
            .filter_map(|point| point.wealth_bins.iter().rposition(|count| *count > 0))
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            first_day,
            last_day,
            maximum,
            last_bin,
        }
    }
    pub fn scale_label(&self, chart: Chart) -> String {
        match chart {
            Chart::Cash => {
                let last = match self.last_bin {
                    63 => "last cash bin >= 2^62".into(),
                    bin => format!("last cash bin {}-{}", 1u64 << (bin - 1), (1u64 << bin) - 1),
                };
                format!(
                    "0-{:.0} households | day {} | {last}",
                    self.maximum, self.last_day
                )
            }
            Chart::Employment => format!(
                "0-{:.0} households / bread ticks | days {}-{}",
                self.maximum, self.first_day, self.last_day
            ),
            Chart::Prices => format!(
                "0-{:.1} cash / goods tick | days {}-{}",
                self.maximum, self.first_day, self.last_day
            ),
            Chart::Wages => format!(
                "0-{:.1} cash / working minute | days {}-{}",
                self.maximum, self.first_day, self.last_day
            ),
            Chart::Inventories => format!(
                "0-{:.0} goods ticks | days {}-{}",
                self.maximum, self.first_day, self.last_day
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_aligns_days_and_stops_at_the_reference_horizon() {
        let reference = ComparisonRun {
            seed: 42,
            horizon: 20,
            history: [18, 19, 20]
                .map(|day| ObservationSnapshot {
                    day,
                    ..Default::default()
                })
                .into(),
        };
        assert!(reference.at_day(17).is_none());
        assert_eq!(reference.at_day(19).unwrap().day, 19);
        assert!(!RunTarget::UntilDay(20).reached(19));
        assert!(RunTarget::UntilDay(20).reached(20));
        assert!(RunTarget::UntilDay(0).reached(0));
        assert!(!RunTarget::Open.reached(20));
    }

    #[test]
    fn chart_scales_use_matching_days_and_current_cash_distribution() {
        let history: VecDeque<ObservationSnapshot> = [
            ObservationSnapshot {
                day: 1,
                employed: 40,
                wealth_bins: vec![0, 100],
                ..Default::default()
            },
            ObservationSnapshot {
                day: 2,
                employed: 20,
                wealth_bins: vec![0, 50, 50],
                ..Default::default()
            },
        ]
        .into();
        let reference = ComparisonRun {
            seed: 4,
            horizon: 3,
            history: [
                ObservationSnapshot {
                    day: 1,
                    employed: 60,
                    wealth_bins: vec![0, 90],
                    ..Default::default()
                },
                ObservationSnapshot {
                    day: 2,
                    employed: 30,
                    wealth_bins: vec![0, 0, 80],
                    ..Default::default()
                },
                ObservationSnapshot {
                    day: 3,
                    employed: 1000,
                    wealth_bins: vec![1000],
                    ..Default::default()
                },
            ]
            .into(),
        };
        let snapshot = history.back().unwrap();
        let employment = ChartRange::new(Chart::Employment, snapshot, &history, Some(&reference));
        assert_eq!(employment.maximum, 60.0);
        assert_eq!(employment.first_day, 1);
        assert_eq!(employment.last_day, 2);
        let cash = ChartRange::new(Chart::Cash, snapshot, &history, Some(&reference));
        assert_eq!(cash.maximum, 80.0);
        assert_eq!(cash.last_bin, 2);
        let mut snapshot = snapshot.clone();
        snapshot.wealth_bins = vec![0; 64];
        snapshot.wealth_bins[63] = 1;
        let cash = ChartRange::new(Chart::Cash, &snapshot, &history, None);
        assert!(cash.scale_label(Chart::Cash).contains(">= 2^62"));
    }
}
