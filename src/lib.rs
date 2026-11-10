pub mod arithmetic;
pub mod domain;
pub mod ledger;
pub mod runtime;
#[cfg(feature = "compute")]
pub mod simulation;
#[cfg(feature = "viewer")]
pub mod viewer;

#[cfg(test)]
extern crate self as bad_econ;
#[cfg(test)]
#[path = "../tests/integration.rs"]
mod tests;
