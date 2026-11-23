#![cfg(feature = "compute")]
mod analytics;
mod bargaining;
mod calendar;
mod contracts;
mod demography;
mod diagnostics;
mod distributors;
mod enterprise;
#[path = "support/estimates.rs"]
mod estimates;
mod execution;
mod finance;
#[path = "support/fixtures.rs"]
mod fixtures;
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
pub(crate) mod support;
mod valuation;
