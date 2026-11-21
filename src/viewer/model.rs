use crate::{
    domain::{Good, ObservationSnapshot, demography::Occupation},
    runtime::RuntimeState,
};
use std::collections::VecDeque;

pub enum DisplayState {
    Initializing,
    Active(RuntimeState),
    Unavailable(String),
}
impl DisplayState {
    pub fn label(&self) -> String {
        match self {
            Self::Initializing => "Initializing simulation...".into(),
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
        self.history
            .binary_search_by_key(&day, |snapshot| snapshot.day)
            .ok()
            .map(|index| &self.history[index])
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Period {
    #[default]
    Month,
    Year,
    FiveYears,
    All,
}
impl Period {
    pub const ALL: [Self; 4] = [Self::Month, Self::Year, Self::FiveYears, Self::All];
    pub fn label(self) -> &'static str {
        match self {
            Self::Month => "30 days",
            Self::Year => "1 year",
            Self::FiveYears => "5 years",
            Self::All => "All history",
        }
    }
    pub fn first_day(self, day: u64) -> u64 {
        match self {
            Self::Month => day.saturating_sub(29),
            Self::Year => day.saturating_sub(364),
            Self::FiveYears => day.saturating_sub(1824),
            Self::All => 0,
        }
    }
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SeriesSelection {
    #[default]
    All,
    Only(usize),
}
impl SeriesSelection {
    pub fn includes(self, series: usize) -> bool {
        match self {
            Self::All => true,
            Self::Only(selected) => series == selected,
        }
    }
    pub fn next(self, count: usize) -> Self {
        match self {
            Self::All => Self::Only(0),
            Self::Only(index) if index + 1 < count => Self::Only(index + 1),
            _ => Self::All,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Chart {
    Employment,
    Unemployment,
    Cash,
    Money,
    Prices,
    Production,
    Consumption,
    Inputs,
    Inventories,
    Transit,
    Shortages,
    Profits,
    Accounts,
    BusinessTypes,
    Industries,
    Businesses,
    BusinessChanges,
    Wages,
    Occupations,
    Population,
    Demography,
    Deaths,
    Housing,
    Credit,
    Debt,
    Rent,
    Arrears,
    Taxes,
    Research,
}

#[derive(Clone, Copy)]
pub(super) enum ValueFormat {
    Money,
    Quantity,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Menu {
    #[default]
    Overview,
    Market,
    Industry,
    Population,
    Finance,
}
impl Menu {
    pub const ALL: [Self; 5] = [
        Self::Overview,
        Self::Market,
        Self::Industry,
        Self::Population,
        Self::Finance,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Overview => "Overview",
            Self::Market => "Market",
            Self::Industry => "Industry",
            Self::Population => "Population",
            Self::Finance => "Finance",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            Self::Overview => "The state of the economy",
            Self::Market => "Goods, prices & supply",
            Self::Industry => "Production & enterprise",
            Self::Population => "People, occupations & living conditions",
            Self::Finance => "Credit & obligations",
        }
    }
    pub fn charts(self) -> &'static [Chart] {
        match self {
            Self::Overview => &[
                Chart::Employment,
                Chart::Unemployment,
                Chart::Cash,
                Chart::Money,
            ],
            Self::Market => &[
                Chart::Prices,
                Chart::Production,
                Chart::Consumption,
                Chart::Inputs,
                Chart::Inventories,
                Chart::Transit,
                Chart::Shortages,
            ],
            Self::Industry => &[
                Chart::Profits,
                Chart::Accounts,
                Chart::BusinessTypes,
                Chart::Industries,
                Chart::Businesses,
                Chart::BusinessChanges,
                Chart::Wages,
                Chart::Research,
            ],
            Self::Population => &[
                Chart::Occupations,
                Chart::Population,
                Chart::Deaths,
                Chart::Demography,
                Chart::Housing,
            ],
            Self::Finance => &[
                Chart::Credit,
                Chart::Debt,
                Chart::Rent,
                Chart::Arrears,
                Chart::Taxes,
            ],
        }
    }
}
impl Chart {
    pub(super) fn value_format(self) -> ValueFormat {
        match self {
            Self::Money
            | Self::Prices
            | Self::Profits
            | Self::Accounts
            | Self::Wages
            | Self::Debt
            | Self::Rent
            | Self::Arrears
            | Self::Taxes => ValueFormat::Money,
            _ => ValueFormat::Quantity,
        }
    }
    pub const ALL: [Self; 29] = [
        Self::Employment,
        Self::Unemployment,
        Self::Cash,
        Self::Money,
        Self::Prices,
        Self::Production,
        Self::Consumption,
        Self::Inputs,
        Self::Inventories,
        Self::Transit,
        Self::Shortages,
        Self::Profits,
        Self::Accounts,
        Self::BusinessTypes,
        Self::Industries,
        Self::Businesses,
        Self::BusinessChanges,
        Self::Wages,
        Self::Occupations,
        Self::Population,
        Self::Demography,
        Self::Deaths,
        Self::Housing,
        Self::Credit,
        Self::Debt,
        Self::Rent,
        Self::Arrears,
        Self::Taxes,
        Self::Research,
    ];
    pub fn tab(self) -> &'static str {
        match self {
            Self::Employment => "Employment",
            Self::Unemployment => "Unemployment",
            Self::Cash => "Cash brackets",
            Self::Money => "Money",
            Self::Prices => "Prices",
            Self::Production => "Production",
            Self::Consumption => "Consumption",
            Self::Inputs => "Inputs",
            Self::Inventories => "Stocks",
            Self::Transit => "Deliveries",
            Self::Shortages => "Food shortage",
            Self::Profits => "Profits",
            Self::Accounts => "Accounts",
            Self::BusinessTypes => "Types",
            Self::Industries => "Industries",
            Self::Businesses => "Lifecycle",
            Self::BusinessChanges => "Entry / exit",
            Self::Wages => "Wages",
            Self::Occupations => "Occupations",
            Self::Population => "Population",
            Self::Demography => "Births / deaths",
            Self::Deaths => "Death causes",
            Self::Housing => "Housing",
            Self::Credit => "Loans",
            Self::Debt => "Debt",
            Self::Rent => "Rent",
            Self::Arrears => "Arrears",
            Self::Taxes => "Taxes",
            Self::Research => "Research",
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Self::Employment => "Labor status",
            Self::Unemployment => "Unemployment rate",
            Self::Cash => "Liquid cash distribution",
            Self::Money => "Spendable cash by sector",
            Self::Prices => "Mean asking prices, rounded to whole units",
            Self::Production => "Goods produced per day",
            Self::Consumption => "Final consumption per day",
            Self::Inputs => "Goods used as inputs per day",
            Self::Inventories => "Available stocks by good",
            Self::Transit => "Goods in transit",
            Self::Shortages => "Unmet daily bread need",
            Self::Profits => "Business cash profit by industry",
            Self::Accounts => "Business receipts, outlays and profit",
            Self::BusinessTypes => "Open businesses by type",
            Self::Industries => "Open business sites by industry",
            Self::Businesses => "Business lifecycle",
            Self::BusinessChanges => "Business entries and closures per day",
            Self::Wages => "Mean offered wages, rounded to whole units",
            Self::Occupations => "Population by occupation",
            Self::Population => "Living population",
            Self::Demography => "Daily demographic change",
            Self::Deaths => "Deaths by cause",
            Self::Housing => "Accounts with insufficient housing",
            Self::Credit => "Active and overdue loans",
            Self::Debt => "Outstanding credit obligations",
            Self::Rent => "Daily rent payments",
            Self::Arrears => "Unpaid rent and payroll reserves",
            Self::Taxes => "Daily tax and employee charges",
            Self::Research => "Research and industrial action",
        }
    }
    pub fn unit(self) -> &'static str {
        match self {
            Self::Employment | Self::Cash => "count",
            Self::Housing => "accounts",
            Self::Unemployment | Self::Occupations => "%",
            Self::Money | Self::Debt | Self::Arrears => "cash",
            Self::Prices => "cash / tick",
            Self::Production | Self::Consumption | Self::Inputs => "ticks / day",
            Self::Inventories | Self::Transit => "ticks",
            Self::Shortages => "bread ticks / day",
            Self::Profits | Self::Accounts | Self::Rent | Self::Taxes => "cash / day",
            Self::BusinessTypes | Self::Industries | Self::Businesses => "business sites",
            Self::BusinessChanges => "sites / day",
            Self::Wages => "cash / minute",
            Self::Population | Self::Deaths => "people",
            Self::Demography => "people / day",
            Self::Credit => "loans",
            Self::Research => "count",
        }
    }
    pub fn description(self) -> &'static str {
        match self {
            Self::Employment => {
                "Paid work counts people who worked a paid shift that day, including farmers. Workers can keep a job through an idle day. Subsistence counts farmers with land. Inactive includes people unavailable for work."
            }
            Self::Unemployment => {
                "Share of unemployed people among wage workers and unemployed people. Available after the first simulated day."
            }
            Self::Cash => {
                "Cohort runs count people by their share of pooled cash. Legacy household runs count accounts by their cash balance."
            }
            Self::Money => {
                "Closing spendable cash. Escrow, payroll reserves and tax treasuries are shown separately from spendable balances."
            }
            Self::Prices => {
                "Mean firm asking prices, not transaction-weighted prices. Missing industries have no observation."
            }
            Self::Production => {
                "Total new output from businesses and households. Computed from changes in cumulative production; inventories and resold goods are not new production."
            }
            Self::Consumption => {
                "Goods consumed by households and other final uses, including expiring housing services. Does not include production inputs."
            }
            Self::Inputs => {
                "Goods used by production, maintenance and construction. Research consumption is tracked separately by the simulation."
            }
            Self::Inventories => {
                "Economy-wide stocks held by households and firms; goods in transit are shown in Deliveries."
            }
            Self::Transit => {
                "Paid goods still traveling to their buyers. Compare with production and stocks to identify delivery bottlenecks."
            }
            Self::Shortages => {
                "Daily unmet household bread requirement, measured in goods ticks, not a count of households."
            }
            Self::Profits => {
                "Cash receipts minus cash outlays between planning and owner distributions. Includes inventory, rent, wages and construction spending; excludes loan funding, debt service and owner payouts. Not accrual profit or GDP."
            }
            Self::Accounts => {
                "Receipts use actual trading-day prices and collected rent. Outlays include wages, inputs and capital purchases. Profit can be negative; no depreciation or inventory revaluation is applied."
            }
            Self::BusinessTypes => {
                "Mutually exclusive types of non-closed business sites. Producers make goods, distributors resell goods, and landlords supply housing."
            }
            Self::Industries => {
                "Counts non-closed sites by the good they supply. Multi-site companies contribute one count per site."
            }
            Self::Businesses => {
                "Current business states; closed firms are excluded. Liquidating firms are winding down."
            }
            Self::BusinessChanges => "Daily changes in cumulative business entries and closures.",
            Self::Wages => {
                "Mean posted wage per working minute. Negotiated or realized worker pay can differ."
            }
            Self::Occupations => {
                "Each living person counts once. Working-age managing owners include landlords. Subsistence farmers retain that occupation when they take paid shifts. Wage workers include commercial farm employees."
            }
            Self::Population => {
                "Living people and age-based dependents. Working age is living population minus dependents, not the number of employed people."
            }
            Self::Demography => {
                "Daily changes in cumulative births and deaths, separated by cause."
            }
            Self::Deaths => {
                "Cumulative deaths since the run began, separated into background mortality, occupational hazards, starvation, and cold exposure."
            }
            Self::Housing => "Population accounts with fewer housing units than they need.",
            Self::Credit => "Current loan counts; overdue loans are a subset of active loans.",
            Self::Debt => {
                "Unpaid principal and currently due interest. These are balances, not daily lending flows."
            }
            Self::Rent => {
                "Change in cumulative rent actually collected. Arrears are tracked separately."
            }
            Self::Arrears => "Outstanding rent balances and cash reserved for payroll settlement.",
            Self::Taxes => "Changes in cumulative tax and employee-charge treasuries.",
            Self::Research => {
                "Research totals and current strike count as recorded by the simulation."
            }
        }
    }
    pub fn menu(self) -> Menu {
        match self {
            Self::Employment | Self::Unemployment | Self::Cash | Self::Money => Menu::Overview,
            Self::Prices
            | Self::Production
            | Self::Consumption
            | Self::Inputs
            | Self::Inventories
            | Self::Transit
            | Self::Shortages => Menu::Market,
            Self::Profits
            | Self::Accounts
            | Self::BusinessTypes
            | Self::Industries
            | Self::Businesses
            | Self::BusinessChanges
            | Self::Wages
            | Self::Research => Menu::Industry,
            Self::Occupations
            | Self::Population
            | Self::Demography
            | Self::Deaths
            | Self::Housing => Menu::Population,
            Self::Credit | Self::Debt | Self::Rent | Self::Arrears | Self::Taxes => Menu::Finance,
        }
    }
    pub fn file_label(self) -> String {
        format!("{self:?}").to_lowercase()
    }
    pub fn next(self) -> Self {
        let index = Self::ALL
            .iter()
            .position(|&chart| chart == self)
            .unwrap_or(0);
        Self::ALL[(index + 1) % Self::ALL.len()]
    }
    pub fn series(self) -> Vec<String> {
        match self {
            Self::Employment => [
                "Wage workers",
                "Unemployed",
                "Farmers with paid work",
                "Farmers without paid work",
                "Inactive",
            ]
            .map(String::from)
            .into(),
            Self::Unemployment => ["No paid work"].map(String::from).into(),
            Self::Cash => [
                "0",
                "1-15",
                "16-255",
                "256-4,095",
                "4,096-65,535",
                "65,536-1,048,575",
                "1,048,576-16,777,215",
                "16,777,216+",
            ]
            .map(String::from)
            .into(),
            Self::Money => ["Population", "Businesses", "Financial institutions"]
                .map(String::from)
                .into(),
            Self::Shortages => ["Unmet bread"].map(String::from).into(),
            Self::Accounts => ["Receipts", "Cash outlays", "Cash profit"]
                .map(String::from)
                .into(),
            Self::BusinessTypes => ["Producers", "Distributors", "Landlords"]
                .map(String::from)
                .into(),
            Self::Businesses => ["Forming", "Operating", "Dormant", "Liquidating"]
                .map(String::from)
                .into(),
            Self::BusinessChanges => ["Entries", "Closures"].map(String::from).into(),
            Self::Occupations => Occupation::ALL
                .into_iter()
                .map(|occupation| occupation.label().into())
                .collect(),
            Self::Population => ["Living", "Dependents", "Working age"]
                .map(String::from)
                .into(),
            Self::Demography => [
                "Births",
                "Background deaths",
                "Work deaths",
                "Famine deaths",
                "Cold deaths",
            ]
            .map(String::from)
            .into(),
            Self::Deaths => ["Background", "Occupational", "Starvation", "Cold"]
                .map(String::from)
                .into(),
            Self::Housing => ["Insufficient housing"].map(String::from).into(),
            Self::Credit => ["Active loans", "Overdue loans"].map(String::from).into(),
            Self::Debt => ["Principal", "Interest due"].map(String::from).into(),
            Self::Rent => ["Rent paid"].map(String::from).into(),
            Self::Arrears => ["Rent arrears", "Payroll reserves"]
                .map(String::from)
                .into(),
            Self::Taxes => ["Income tax", "Employee charges"].map(String::from).into(),
            Self::Research => ["Attempts", "Discoveries", "Striking"]
                .map(String::from)
                .into(),
            _ => Good::ALL.into_iter().map(|good| good.to_string()).collect(),
        }
    }
    pub(super) fn samples(
        self,
        history: &VecDeque<ObservationSnapshot>,
        first_day: u64,
        last_day: u64,
    ) -> impl Iterator<Item = ChartSample> {
        let start = history.partition_point(|point| point.day < first_day);
        history
            .range(start..)
            .take_while(move |point| point.day <= last_day)
            .enumerate()
            .map(move |(offset, snapshot)| {
                let previous = (start + offset).checked_sub(1).map(|index| &history[index]);
                ChartSample {
                    day: snapshot.day,
                    values: self.values(snapshot, previous),
                }
            })
    }

    pub fn values(
        self,
        s: &ObservationSnapshot,
        previous: Option<&ObservationSnapshot>,
    ) -> Vec<f64> {
        match self {
            Self::Employment => [
                s.economy.wage_employed(s.employed) as f64,
                s.economy
                    .workforce
                    .saturating_sub(s.economy.wage_employed(s.employed)) as f64,
                s.economy.rural_employed as f64,
                s.economy
                    .subsistence
                    .saturating_sub(s.economy.rural_employed) as f64,
                s.economy.inactive as f64,
            ]
            .map(|value| daily_value(s, value))
            .into(),
            Self::Unemployment => vec![
                s.population
                    .unemployment_rate()
                    .filter(|_| s.day > 0)
                    .unwrap_or(f64::NAN),
            ],
            Self::Cash => [0..1, 1..5, 5..9, 9..13, 13..17, 17..21, 21..25, 25..64]
                .into_iter()
                .map(|bins| {
                    s.wealth_bins
                        .iter()
                        .skip(bins.start)
                        .take(bins.len())
                        .map(|&count| count as f64)
                        .sum()
                })
                .collect(),
            Self::Money => vec![
                s.economy.household_cash as f64,
                s.economy.firm_cash.iter().map(|&v| v as f64).sum(),
                s.financial_cash as f64,
            ],
            Self::Prices => s
                .markets
                .iter()
                .map(|m| match m.prices.count {
                    0 => f64::NAN,
                    _ => m.prices.mean(),
                })
                .collect(),
            Self::Production => daily_goods(s, previous, |s| &s.produced),
            Self::Consumption => daily_goods(s, previous, |s| &s.consumption),
            Self::Inputs => daily_goods(s, previous, |s| &s.used),
            Self::Inventories => s.inventories.iter().map(|&v| v as f64).collect(),
            Self::Transit => s.transit.iter().map(|&v| v as f64).collect(),
            Self::Shortages => vec![s.shortages as f64],
            Self::Profits => s
                .economy
                .accounts
                .iter()
                .map(|a| daily_value(s, a.profit() as f64))
                .collect(),
            Self::Accounts => vec![
                daily_value(s, s.economy.accounts.iter().map(|a| a.revenue as f64).sum()),
                daily_value(
                    s,
                    s.economy.accounts.iter().map(|a| a.outlays() as f64).sum(),
                ),
                daily_value(
                    s,
                    s.economy.accounts.iter().map(|a| a.profit() as f64).sum(),
                ),
            ],
            Self::BusinessTypes => s.economy.business_types.iter().map(|&v| v as f64).collect(),
            Self::Industries => s.economy.active_by_good.iter().map(|&v| v as f64).collect(),
            Self::Businesses => s
                .economy
                .business_states
                .iter()
                .map(|&v| v as f64)
                .collect(),
            Self::BusinessChanges => vec![
                daily_change(s, previous, |s| s.firm_entries),
                daily_change(s, previous, |s| s.firm_closures),
            ],
            Self::Wages => s
                .markets
                .iter()
                .map(|m| match m.wages.count {
                    0 => f64::NAN,
                    _ => m.wages.mean(),
                })
                .collect(),
            Self::Occupations => s
                .population
                .occupations
                .iter()
                .map(|&v| match s.population.living {
                    0 => f64::NAN,
                    n => daily_value(s, 100.0 * v as f64 / n as f64),
                })
                .collect(),
            Self::Population => vec![
                s.population.living as f64,
                s.population.dependents as f64,
                s.population.living.saturating_sub(s.population.dependents) as f64,
            ],
            Self::Demography => vec![
                daily_change(s, previous, |s| s.population.births),
                daily_change(s, previous, |s| s.population.deaths[0]),
                daily_change(s, previous, |s| s.population.deaths[1]),
                daily_change(s, previous, |s| s.population.deaths[2]),
                daily_change(s, previous, |s| s.population.deaths[3]),
            ],
            Self::Deaths => s
                .population
                .deaths
                .iter()
                .map(|&deaths| deaths as f64)
                .collect(),
            Self::Housing => vec![s.homeless_households as f64],
            Self::Credit => vec![s.active_loans as f64, s.overdue_loans as f64],
            Self::Debt => vec![s.loan_principal as f64, s.loan_interest as f64],
            Self::Rent => vec![daily_change(s, previous, |s| s.rent_collected)],
            Self::Arrears => vec![s.rent_arrears as f64, s.payroll_reserves as f64],
            Self::Taxes => vec![
                daily_change(s, previous, |s| s.income_tax),
                daily_change(s, previous, |s| s.employee_charges),
            ],
            Self::Research => vec![
                s.research.attempts as f64,
                s.research.discoveries as f64,
                s.research.striking as f64,
            ],
        }
    }
}
fn daily_value(snapshot: &ObservationSnapshot, value: f64) -> f64 {
    match snapshot.day {
        0 => f64::NAN,
        _ => value,
    }
}
fn daily_change(
    snapshot: &ObservationSnapshot,
    previous: Option<&ObservationSnapshot>,
    field: impl Fn(&ObservationSnapshot) -> u64,
) -> f64 {
    previous
        .filter(|prior| snapshot.day > prior.day)
        .and_then(|prior| {
            field(snapshot)
                .checked_sub(field(prior))
                .map(|change| change as f64 / (snapshot.day - prior.day) as f64)
        })
        .unwrap_or(f64::NAN)
}
fn daily_goods(
    snapshot: &ObservationSnapshot,
    previous: Option<&ObservationSnapshot>,
    field: impl Fn(&ObservationSnapshot) -> &[u64; Good::COUNT],
) -> Vec<f64> {
    (0..Good::COUNT)
        .map(|good| daily_change(snapshot, previous, |s| field(s)[good]))
        .collect()
}

pub(super) struct ChartSample {
    pub day: u64,
    pub values: Vec<f64>,
}

pub struct ChartRange {
    pub first_day: u64,
    pub last_day: u64,
    pub minimum: f64,
    pub maximum: f64,
    pub selection: SeriesSelection,
}
impl ChartRange {
    pub fn new(
        chart: Chart,
        snapshot: &ObservationSnapshot,
        history: &VecDeque<ObservationSnapshot>,
        comparison: Option<&ComparisonRun>,
        period: Period,
        selection: SeriesSelection,
    ) -> Self {
        let first_day = history
            .front()
            .map_or(snapshot.day, |point| point.day)
            .max(period.first_day(snapshot.day));
        let last_day = snapshot.day;
        let range = std::iter::once(history)
            .chain(comparison.map(|run| &run.history))
            .flat_map(|history| chart.samples(history, first_day, last_day))
            .flat_map(|sample| sample.values.into_iter().enumerate())
            .filter_map(|(series, value)| {
                (selection.includes(series) && value.is_finite()).then_some(value)
            })
            .fold(
                Self {
                    first_day,
                    last_day,
                    minimum: 0.0,
                    maximum: 1.0,
                    selection,
                },
                |range, value| Self {
                    minimum: range.minimum.min(value),
                    maximum: range.maximum.max(value),
                    ..range
                },
            );
        Self {
            maximum: match chart {
                Chart::Unemployment | Chart::Occupations => 100.0,
                _ => range.maximum,
            },
            ..range
        }
    }
    pub fn fraction(&self, value: f64) -> f64 {
        (value - self.minimum) / (self.maximum - self.minimum)
    }
    pub fn scale_label(&self, chart: Chart) -> String {
        format!(
            "{} | days {}-{} | hover for exact values",
            chart.unit(),
            self.first_day,
            self.last_day
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chart_scales_use_matching_days_and_cash_bracket_history() {
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
        let employment = ChartRange::new(
            Chart::Employment,
            snapshot,
            &history,
            Some(&reference),
            Period::All,
            SeriesSelection::All,
        );
        assert_eq!(employment.maximum, 60.0);
        assert_eq!(employment.first_day, 1);
        assert_eq!(employment.last_day, 2);
        let cash = ChartRange::new(
            Chart::Cash,
            snapshot,
            &history,
            Some(&reference),
            Period::All,
            SeriesSelection::All,
        );
        assert_eq!(cash.maximum, 100.0);
    }
    #[test]
    fn periods_clip_levels_but_keep_previous_day_for_daily_flows() {
        let history: VecDeque<_> = (0..=2000)
            .map(|day| {
                let mut point = ObservationSnapshot {
                    day,
                    ..Default::default()
                };
                point.produced[0] = day * 7;
                point.inventories[0] = match day {
                    0 => 1_000_000,
                    _ => 10,
                };
                point
            })
            .collect();
        let snapshot = history.back().unwrap();
        for (period, first) in [
            (Period::Month, 1971),
            (Period::Year, 1636),
            (Period::FiveYears, 176),
            (Period::All, 0),
        ] {
            let range = ChartRange::new(
                Chart::Inventories,
                snapshot,
                &history,
                None,
                period,
                SeriesSelection::All,
            );
            assert_eq!(range.first_day, first);
            assert_eq!(range.last_day, 2000);
            assert_eq!(range.maximum, if first == 0 { 1_000_000.0 } else { 10.0 });
        }
        let range = ChartRange::new(
            Chart::Production,
            snapshot,
            &history,
            None,
            Period::Month,
            SeriesSelection::All,
        );
        assert_eq!(range.maximum, 7.0);
        assert_eq!(
            Chart::Production.values(&history[1971], Some(&history[1970]))[0],
            7.0
        );
        assert!(Chart::Production.values(&history[0], None)[0].is_nan());
    }

    #[test]
    fn losses_and_focused_series_use_signed_ranges() {
        let mut snapshot = ObservationSnapshot {
            day: 1,
            ..Default::default()
        };
        snapshot.economy.accounts[0] = crate::domain::diagnostics::BusinessAccount {
            opening_cash: 300,
            closing_cash: 100,
            revenue: 50,
        };
        snapshot.economy.accounts[1] = crate::domain::diagnostics::BusinessAccount {
            opening_cash: 100,
            closing_cash: 1100,
            revenue: 1000,
        };
        let history = VecDeque::from([snapshot.clone()]);
        let range = ChartRange::new(
            Chart::Profits,
            &snapshot,
            &history,
            None,
            Period::All,
            SeriesSelection::All,
        );
        assert_eq!(range.minimum, -200.0);
        assert_eq!(range.maximum, 1000.0);
        assert_eq!(range.fraction(-200.0), 0.0);
        assert_eq!(range.fraction(1000.0), 1.0);
        let range = ChartRange::new(
            Chart::Profits,
            &snapshot,
            &history,
            None,
            Period::All,
            SeriesSelection::Only(0),
        );
        assert_eq!(range.minimum, -200.0);
        assert_eq!(range.maximum, 1.0);
    }

    #[test]
    fn population_shares_and_empty_workforce_have_explicit_meanings() {
        let mut snapshot = ObservationSnapshot {
            day: 1,
            ..Default::default()
        };
        snapshot.population.living = 20;
        snapshot.population.occupations = [4, 5, 2, 1, 6, 2];
        assert_eq!(
            Chart::Occupations.values(&snapshot, None),
            vec![20.0, 25.0, 10.0, 5.0, 30.0, 10.0]
        );
        assert!((Chart::Unemployment.values(&snapshot, None)[0] - 100.0 / 6.0).abs() < 0.000_001);
        snapshot.population.occupations = [0; 6];
        assert!(Chart::Unemployment.values(&snapshot, None)[0].is_nan());
        snapshot.economy.workforce = 10;
        snapshot.employed = 7;
        snapshot.economy.rural_employed = 3;
        snapshot.economy.subsistence = 8;
        snapshot.economy.inactive = 2;
        assert_eq!(
            Chart::Employment.values(&snapshot, None),
            vec![4.0, 6.0, 3.0, 5.0, 2.0]
        );
        for chart in Chart::ALL {
            assert_eq!(chart.series().len(), chart.values(&snapshot, None).len());
            assert!(chart.menu().charts().contains(&chart));
        }
    }
}
