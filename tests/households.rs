use super::support;
use anyhow::Result;
use bad_econ::{arithmetic::DrawKey, simulation::households::*};

fn observed(prior_forecast: u64, signal: u64, observation_weight: u32) -> HouseholdObservation {
    let draw = DrawKey {
        seed: signal,
        day: 0,
        phase: 0,
        actor: prior_forecast,
        purpose: observation_weight,
        index: 0,
    }
    .draw();
    HouseholdObservation {
        services: vec![
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
        ],
        participation: Participation::Available,
        capacity_minutes: 480,
        fatigue: 200,
        reserved_minutes: 960,
        fatigue_response: 100,
        exertion_minutes: 80,
        rest_minutes: 480,
        prior_forecast,
        observation: Observation::Observed(signal),
        observation_weight,
    }
}

#[test]
fn household_deficits_and_forecasts_preserve_full_width_rounding() -> Result<()> {
    let device = support::device(64)?;
    let extremes = [0, 1, u64::MAX / 2, u64::MAX];
    let cases = [0, 1, 500, 999, 1000]
        .into_iter()
        .flat_map(|weight| {
            extremes
                .into_iter()
                .flat_map(move |prior| extremes.map(|signal| observed(prior, signal, weight)))
        })
        .collect::<Vec<_>>();
    let results = evaluate(&device, &cases)?;
    assert_eq!(results.len(), cases.len());
    for (case, result) in cases.iter().zip(results) {
        let mut lower = 0;
        let mut upper = 0;
        for service in case.services.iter().filter(|service| service.need > 0) {
            let need = u128::from(service.need);
            let deficit = u128::from(service.weight)
                * u128::from(service.need.saturating_sub(service.delivered));
            lower += deficit / need;
            upper += deficit.div_ceil(need);
        }
        assert_eq!(u128::from(result.deficit_lower), lower);
        assert_eq!(u128::from(result.deficit_upper), upper);
        assert!(result.deficit_upper <= HOUSEHOLD_SCALE);
        let signal = match case.observation {
            Observation::Observed(signal) => signal,
            Observation::Missing => case.prior_forecast,
        };
        let expected = (u128::from(case.prior_forecast)
            * u128::from(1000 - case.observation_weight)
            + u128::from(signal) * u128::from(case.observation_weight))
            / 1000;
        assert_eq!(u128::from(result.forecast), expected);
    }
    let empty = HouseholdObservation {
        services: Vec::new(),
        ..observed(0, 0, 0)
    };
    assert_eq!(evaluate(&device, &[empty])?[0].deficit_lower, 0);
    Ok(())
}
