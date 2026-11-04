#![cfg(feature = "compute")]
use crate::support;

use anyhow::{Result, anyhow};
use bad_econ::simulation::{analytics::*, research::capabilities::CapabilityEdge};

fn matrix(rows: usize, columns: usize, values: &[u32]) -> Result<SparseMatrix> {
    SparseMatrix::new(
        rows,
        columns,
        values
            .iter()
            .enumerate()
            .map(|(index, &numerator)| Coefficient {
                row: index / columns,
                column: index % columns,
                numerator,
                denominator: 1,
            })
            .collect(),
    )
}

fn request() -> Result<AnalysisRequest> {
    Ok(AnalysisRequest {
        context: ContextVersion {
            techniques: 7,
            classification: 3,
            capability_graph: 5,
            ..ContextVersion::default()
        },
        scale: 1000,
        goods: matrix(3, 3, &[0, 1, 0, 0, 0, 1, 0, 0, 0])?,
        labor: matrix(2, 3, &[1, 0, 0, 0, 2, 3])?,
        weights: vec![1000, 1000],
        final_demand: vec![0, 0, 1000],
        goods_certificate: Contraction::new(1, 2, vec![4, 2, 1])?,
        value_certificate: Contraction::new(1, 2, vec![1, 2, 4])?,
        budget: SolveBudget {
            iterations: 3,
            weighted_tolerance: 0,
        },
        reference_occupation: 0,
        research: None,
        reproduction: None,
        comparator: None,
        valuation: Some(ValuationRequest {
            activities: vec![2, 0, 1],
            capabilities: Some(CapabilityInputs {
                labels: vec![1, 2, 4],
                dependencies: CapabilityDependencies::InputSupport,
            }),
        }),
    })
}

fn records(report: AnalysisReport) -> Result<Vec<ValuationRecord>> {
    match report.valuations {
        Diagnostic::Available(records) => Ok(records),
        diagnostic => Err(anyhow!("valuation unavailable: {diagnostic:?}")),
    }
}

#[test]
fn lean_valuation_keeps_context_raw_hours_and_support_together() -> Result<()> {
    let device = support::device(64)?;
    let mut request = request()?;
    let serialized = serde_json::to_string(&request)?;
    let restored: AnalysisRequest = serde_json::from_str(&serialized)?;
    let initial = records(restored.run(&device)?)?;
    for (record, (activity, hours, labels)) in initial.iter().zip([
        (2, [1000, 5000], 7),
        (0, [1000, 0], 1),
        (1, [1000, 2000], 3),
    ]) {
        assert_eq!(record.context, request.context);
        assert_eq!(record.activity, activity);
        assert_eq!(record.scale, 1000);
        assert_eq!(
            record.occupational_hours,
            Diagnostic::Available(
                hours
                    .into_iter()
                    .map(|value| Interval {
                        lower: value,
                        upper: value
                    })
                    .collect()
            )
        );
        assert_eq!(
            record.supporting_capabilities,
            Diagnostic::Available(labels)
        );
    }
    request.weights = vec![3000, 500];
    request.research = Some(ResearchRequest {
        inputs: matrix(3, 1, &[1, 0, 0])?,
        hours: matrix(2, 1, &[100, 100])?,
        attempts: vec![1000],
    });
    assert_eq!(records(request.run(&device)?)?, initial);
    request.context.capability_graph += 1;
    request
        .valuation
        .as_mut()
        .unwrap()
        .capabilities
        .as_mut()
        .unwrap()
        .dependencies = CapabilityDependencies::Declared(vec![
        CapabilityEdge {
            prerequisite: 0,
            dependent: 1,
        },
        CapabilityEdge {
            prerequisite: 1,
            dependent: 2,
        },
        CapabilityEdge {
            prerequisite: 2,
            dependent: 0,
        },
    ]);
    let declared = records(request.run(&device)?)?;
    for (record, before) in declared.iter().zip(&initial) {
        assert_eq!(record.context, request.context);
        assert_eq!(record.occupational_hours, before.occupational_hours);
        assert_eq!(record.supporting_capabilities, Diagnostic::Available(7));
    }
    Ok(())
}

#[test]
fn missing_labels_and_unavailable_hours_have_independent_diagnostics() -> Result<()> {
    let device = support::device(64)?;
    let mut request = request()?;
    request.budget.iterations = 0;
    let unresolved = records(request.run(&device)?)?;
    assert!(unresolved.iter().all(|record| {
        matches!(record.occupational_hours, Diagnostic::UnresolvedAccuracy)
            && matches!(record.supporting_capabilities, Diagnostic::Available(_))
    }));
    request.goods = matrix(3, 3, &[1, 0, 0, 0, 1, 0, 0, 0, 1])?;
    let inadmissible = records(request.run(&device)?)?;
    assert!(inadmissible.iter().all(|record| {
        matches!(record.occupational_hours, Diagnostic::Inadmissible)
            && matches!(record.supporting_capabilities, Diagnostic::Available(_))
    }));
    request.valuation.as_mut().unwrap().capabilities = None;
    assert!(
        records(request.run(&device)?)?
            .iter()
            .all(|record| { matches!(record.supporting_capabilities, Diagnostic::MissingData) })
    );
    request.valuation.as_mut().unwrap().capabilities = Some(CapabilityInputs {
        labels: vec![0, 0, 0],
        dependencies: CapabilityDependencies::InputSupport,
    });
    assert!(
        records(request.run(&device)?)?
            .iter()
            .all(|record| { matches!(record.supporting_capabilities, Diagnostic::Available(0)) })
    );
    request.valuation = None;
    assert!(matches!(
        request.run(&device)?.valuations,
        Diagnostic::Inapplicable
    ));
    Ok(())
}
