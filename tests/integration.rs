mod analytics;
mod bargaining;
mod calendar;
mod contracts;
mod demography;
mod diagnostics;
mod distributors;
mod enterprise;
#[cfg(feature = "compute")]
#[path = "support/estimates.rs"]
mod estimates;
mod execution;
mod finance;
mod food;
mod heating;
mod households;
mod industry;
mod institutions;
mod landlords;
mod payroll;
mod policies;
mod population;
mod pricing;
mod property;
mod railway;
mod rent;
mod research;
mod retained;
mod seasonal_farming;
mod sensitivity;
#[cfg(feature = "compute")]
pub(crate) mod support;
mod valuation;
