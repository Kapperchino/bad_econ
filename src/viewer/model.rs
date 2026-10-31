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
            Self::Initializing => "Initializing shared GPU…".into(),
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
            Self::Prices => "Mean prices · grain / flour / bread",
            Self::Wages => "Mean wages · grain / flour / bread",
            Self::Inventories => "Available stocks · grain / flour / bread",
            Self::Cash => "Household cash · bins 0, 1, 2–3, 4–7, …",
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
}
