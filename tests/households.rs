#![cfg(feature = "compute")]
use super::support;

use anyhow::Result;
use bad_econ::simulation::households::*;

fn observed() -> HouseholdObservation {
    HouseholdObservation {
        services: vec![
            DeliveredService {
                need: 2,
                delivered: 1,
                weight: 500,
            },
            DeliveredService {
                need: 1,
                delivered: 1,
                weight: 500,
            },
        ],
        participation: Participation::Available,
        capacity_minutes: 480,
        fatigue: 200,
        reserved_minutes: 960,
        fatigue_response: 100,
        exertion_minutes: 80,
        rest_minutes: 480,
        prior_forecast: 10,
        observation: Observation::Observed(14),
        observation_weight: 250,
    }
}

#[test]
fn lean_weighted_need_missing_observation_and_shared_clock() -> Result<()> {
    let device = support::device(64)?;
    let mut absent = observed();
    absent.observation = Observation::Missing;
    absent.participation = Participation::Unavailable;
    let mut over_reserved = observed();
    over_reserved.reserved_minutes = 2000;
    let actual = evaluate(&device, &[observed(), absent, over_reserved])?;
    assert_eq!(
        actual[0],
        HouseholdPolicyResult {
            deficit_lower: 250,
            deficit_upper: 250,
            available_minutes: 384,
            fatigue: 116,
            forecast: 11,
        }
    );
    assert_eq!(actual[1].available_minutes, 0);
    assert_eq!(actual[1].forecast, 10);
    assert_eq!(actual[2].available_minutes, 0);
    Ok(())
}

#[test]
fn weighted_deficits_keep_zero_needs_and_full_width_deliveries_exact() -> Result<()> {
    let device = support::device(64)?;
    let cases = (0..64u64)
        .map(|index| {
            let draw = bad_econ::arithmetic::DrawKey {
                seed: 601,
                day: 0,
                phase: 0,
                actor: index,
                purpose: 0,
                index: 0,
            }
            .draw();
            let mut request = observed();
            request.services = vec![
                DeliveredService {
                    need: draw.max(1),
                    delivered: draw.rotate_left(11),
                    weight: 400,
                },
                DeliveredService {
                    need: u64::MAX,
                    delivered: draw,
                    weight: 600,
                },
                DeliveredService {
                    need: 0,
                    delivered: u64::MAX,
                    weight: u32::MAX,
                },
            ];
            request
        })
        .collect::<Vec<_>>();
    let actual = evaluate(&device, &cases)?;
    for (request, result) in cases.iter().zip(actual) {
        let expected_lower: u128 = request
            .services
            .iter()
            .filter(|service| service.need > 0)
            .map(|service| {
                u128::from(service.weight)
                    * u128::from(service.need.saturating_sub(service.delivered))
                    / u128::from(service.need)
            })
            .sum();
        let expected_upper: u128 = request
            .services
            .iter()
            .filter(|service| service.need > 0)
            .map(|service| {
                (u128::from(service.weight)
                    * u128::from(service.need.saturating_sub(service.delivered)))
                .div_ceil(u128::from(service.need))
            })
            .sum();
        assert_eq!(u128::from(result.deficit_lower), expected_lower);
        assert_eq!(u128::from(result.deficit_upper), expected_upper);
        assert!(result.deficit_upper <= HOUSEHOLD_SCALE);
    }
    let mut empty = observed();
    empty.services.clear();
    assert_eq!(evaluate(&device, &[empty])?[0].deficit_lower, 0);
    Ok(())
}

#[test]
fn convex_blend_rounding_preserves_endpoints_and_does_not_overflow() -> Result<()> {
    let device = support::device(64)?;
    let cases = [0, 1, 500, 999, 1000]
        .into_iter()
        .flat_map(|weight| {
            [0, 1, u64::MAX / 2, u64::MAX]
                .into_iter()
                .flat_map(move |old| {
                    [0, 1, u64::MAX / 2, u64::MAX]
                        .into_iter()
                        .map(move |signal| {
                            let mut request = observed();
                            request.prior_forecast = old;
                            request.observation = Observation::Observed(signal);
                            request.observation_weight = weight;
                            request
                        })
                })
        })
        .collect::<Vec<_>>();
    for (request, result) in cases.iter().zip(evaluate(&device, &cases)?) {
        let signal = match request.observation {
            Observation::Observed(signal) => signal,
            Observation::Missing => 0,
        };
        let expected = (u128::from(request.prior_forecast)
            * u128::from(1000 - request.observation_weight)
            + u128::from(signal) * u128::from(request.observation_weight))
            / 1000;
        assert_eq!(u128::from(result.forecast), expected);
    }
    Ok(())
}
