use super::*;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResearchRequest {
    pub inputs: SparseMatrix,
    pub hours: SparseMatrix,
    pub attempts: Vec<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReproductionRequest {
    pub goods: SparseMatrix,
    pub training: SparseMatrix,
    pub certificate: Contraction,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparatorRequest {
    pub context: ComparatorContext,
    pub living: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisRequest {
    pub context: ContextVersion,
    pub scale: u32,
    pub goods: SparseMatrix,
    pub labor: SparseMatrix,
    pub weights: Vec<u64>,
    pub final_demand: Vec<u64>,
    pub goods_certificate: Contraction,
    pub value_certificate: Contraction,
    pub budget: SolveBudget,
    pub reference_occupation: usize,
    #[serde(default)]
    pub research: Option<ResearchRequest>,
    #[serde(default)]
    pub reproduction: Option<ReproductionRequest>,
    #[serde(default)]
    pub comparator: Option<ComparatorRequest>,
    #[serde(default)]
    pub valuation: Option<ValuationRequest>,
}

#[derive(Clone, Debug, Serialize)]
pub struct AnalysisReport {
    pub context: ContextVersion,
    pub profile: &'static str,
    pub scale: u32,
    pub goods_values: Diagnostic<ApproximateValues>,
    pub gross_requirements: Diagnostic<ApproximateValues>,
    pub occupational_hours: Diagnostic<Vec<Interval>>,
    pub research_values: Diagnostic<Vec<Interval>>,
    pub plan_value: Diagnostic<Vec<Interval>>,
    pub coupled_values: Diagnostic<ApproximateValues>,
    pub normalized_coupled: Diagnostic<Vec<Interval>>,
    pub residual_scale: Diagnostic<Vec<Interval>>,
    pub comparator: Diagnostic<ComparatorReport>,
    pub valuations: Diagnostic<Vec<ValuationRecord>>,
}

impl AnalysisRequest {
    pub fn run(&self, device: &Device) -> Result<AnalysisReport> {
        let model = AccountModel::new(self.goods.clone(), self.labor.clone())?;
        let model = match &self.research {
            Some(research) => {
                model.with_research(research.inputs.clone(), research.hours.clone())?
            }
            None => model,
        };
        let model = match &self.reproduction {
            Some(reproduction) => model
                .with_reproduction(reproduction.goods.clone(), reproduction.training.clone())?,
            None => model,
        };
        let admitted = self.scale > 0
            && self.reference_occupation < self.labor.rows()
            && self.weights.len() == self.labor.rows()
            && self.final_demand.len() == self.goods.rows()
            && self
                .research
                .as_ref()
                .is_none_or(|research| research.attempts.len() == research.inputs.columns());
        match admitted {
            false => Err(anyhow!(
                "analysis vectors, reference occupation, or scale disagree with the model"
            )),
            true => {
                let account = model.upload(device)?;
                let weights = DeviceVector::from_ticks(device, &self.weights)?;
                let final_demand = DeviceVector::from_ticks(device, &self.final_demand)?;
                let values = account.external_values(
                    device,
                    &weights,
                    &self.value_certificate,
                    self.budget,
                )?;
                let attempts = self
                    .research
                    .as_ref()
                    .map(|research| DeviceVector::from_ticks(device, &research.attempts))
                    .transpose()?;
                let plan = account.production_plan(
                    device,
                    &final_demand,
                    attempts.as_ref(),
                    &self.goods_certificate,
                    self.budget,
                )?;
                let hours = plan
                    .occupational_hours
                    .try_map(|hours| hours.read(device))?;
                let available_values = values.values(device)?;
                let research_values = available_values
                    .try_and_then(|values| account.research_values(device, values, &weights))?;
                let plan_value = available_values.try_and_then(|values| {
                    let ordinary = values.dot(device, &final_demand, self.scale)?;
                    Ok(match (&attempts, &research_values) {
                        (Some(attempts), Diagnostic::Available(costs)) => Diagnostic::Available(
                            ordinary.add(device, &costs.dot(device, attempts, self.scale)?)?,
                        ),
                        (None, _) => Diagnostic::Available(ordinary),
                        (_, Diagnostic::Inadmissible) => Diagnostic::Inadmissible,
                        _ => Diagnostic::UnresolvedAccuracy,
                    })
                })?;
                let coupled = match &self.reproduction {
                    Some(reproduction) => account.coupled_values(
                        device,
                        self.scale,
                        &reproduction.certificate,
                        self.budget,
                    )?,
                    None => Diagnostic::Inapplicable,
                };
                let available_coupled = coupled.try_and_then(|solution| solution.values(device))?;
                let reference = self.goods.rows() + self.reference_occupation;
                let normalized = available_coupled
                    .try_and_then(|values| values.normalize(device, reference, self.scale))?;
                let comparator = match (&self.comparator, &plan_value) {
                    (Some(request), Diagnostic::Available(total)) => {
                        Diagnostic::Available(comparator(
                            device,
                            request.context.clone(),
                            request.living,
                            total,
                            self.scale,
                        )?)
                    }
                    (None, _) => Diagnostic::Inapplicable,
                    (_, Diagnostic::Inadmissible) => Diagnostic::Inadmissible,
                    _ => Diagnostic::UnresolvedAccuracy,
                };
                let residual_scale = available_coupled.try_and_then(|values| {
                    values.reciprocal_reference(device, reference, self.scale)
                })?;
                Ok(AnalysisReport {
                    context: self.context,
                    profile: ANALYTICAL_PROFILE,
                    scale: self.scale,
                    goods_values: values.read(device)?,
                    gross_requirements: plan.gross_requirements.read(device)?,
                    occupational_hours: hours,
                    research_values: research_values.try_map(|values| values.read(device))?,
                    plan_value: plan_value.try_map(|value| value.read(device))?,
                    coupled_values: coupled.try_and_then(|solution| solution.read(device))?,
                    normalized_coupled: normalized.try_map(|values| values.read(device))?,
                    residual_scale: residual_scale.try_map(|value| value.read(device))?,
                    comparator,
                    valuations: match &self.valuation {
                        Some(request) => {
                            Diagnostic::Available(request.records(device, self, &account)?)
                        }
                        None => Diagnostic::Inapplicable,
                    },
                })
            }
        }
    }
}
