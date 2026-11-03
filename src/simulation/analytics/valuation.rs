use super::*;
use crate::simulation::research::capabilities::{CapabilityEdge, CapabilityGraph};
use std::collections::BTreeSet;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValuationRequest {
    pub activities: Vec<usize>,
    #[serde(default)]
    pub capabilities: Option<CapabilityInputs>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapabilityInputs {
    pub labels: Vec<u32>,
    pub dependencies: CapabilityDependencies,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CapabilityDependencies {
    InputSupport,
    Declared(Vec<CapabilityEdge>),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ValuationRecord {
    pub context: ContextVersion,
    pub activity: usize,
    pub scale: u32,
    pub occupational_hours: Diagnostic<Vec<Interval>>,
    pub supporting_capabilities: Diagnostic<u32>,
}

impl CapabilityInputs {
    fn closure(&self, device: &Device, goods: &SparseMatrix) -> Result<Vec<u32>> {
        let edges = match &self.dependencies {
            CapabilityDependencies::InputSupport => goods
                .entries()
                .iter()
                .filter(|entry| entry.numerator > 0)
                .map(|entry| CapabilityEdge {
                    prerequisite: entry.row as u32,
                    dependent: entry.column as u32,
                })
                .collect(),
            CapabilityDependencies::Declared(edges) => edges.clone(),
        };
        CapabilityGraph::new(goods.rows(), edges)?
            .closure(device, &self.labels)?
            .download(device)
    }
}

impl ValuationRequest {
    pub(super) fn records(
        &self,
        device: &Device,
        analysis: &AnalysisRequest,
        account: &DeviceAccounts,
    ) -> Result<Vec<ValuationRecord>> {
        let activities = self.activities.iter().copied().collect::<BTreeSet<_>>();
        match activities.len() == self.activities.len()
            && activities
                .iter()
                .all(|&activity| activity < analysis.goods.rows())
        {
            true => Ok(()),
            false => Err(anyhow!(
                "valuation activities must be distinct goods indices"
            )),
        }?;
        let capabilities = self
            .capabilities
            .as_ref()
            .map(|inputs| inputs.closure(device, &analysis.goods))
            .transpose()?;
        self.activities
            .iter()
            .map(|&activity| {
                let mut demand = vec![0; analysis.goods.rows()];
                demand[activity] = u64::from(analysis.scale);
                let solution = account.goods_requirements(
                    device,
                    &DeviceVector::from_ticks(device, &demand)?,
                    &analysis.goods_certificate,
                    analysis.budget,
                )?;
                let occupational_hours = match solution.values(device)? {
                    Diagnostic::Available(requirements) => Diagnostic::Available(
                        account
                            .occupational_hours(device, requirements)?
                            .read(device)?,
                    ),
                    Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                    _ => Diagnostic::UnresolvedAccuracy,
                };
                Ok(ValuationRecord {
                    context: analysis.context,
                    activity,
                    scale: analysis.scale,
                    occupational_hours,
                    supporting_capabilities: match &capabilities {
                        Some(labels) => Diagnostic::Available(labels[activity]),
                        None => Diagnostic::MissingData,
                    },
                })
            })
            .collect()
    }
}
