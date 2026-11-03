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
                let demand = match &self.research {
                    Some(research) => final_demand.add(
                        device,
                        &DeviceOperator::new(device, &research.inputs).apply(
                            device,
                            &DeviceVector::from_ticks(device, &research.attempts)?,
                        )?,
                    )?,
                    None => DeviceVector {
                        data: final_demand.data.clone(),
                        length: final_demand.length,
                    },
                };
                let gross = account.goods_requirements(
                    device,
                    &demand,
                    &self.goods_certificate,
                    self.budget,
                )?;
                let hours = match gross.values(device)? {
                    Diagnostic::Available(gross) => {
                        let production =
                            DeviceOperator::new(device, &self.labor).apply(device, gross)?;
                        let hours = match &self.research {
                            Some(research) => production.add(
                                device,
                                &DeviceOperator::new(device, &research.hours).apply(
                                    device,
                                    &DeviceVector::from_ticks(device, &research.attempts)?,
                                )?,
                            )?,
                            None => production,
                        };
                        Diagnostic::Available(hours.read(device)?)
                    }
                    Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                    _ => Diagnostic::UnresolvedAccuracy,
                };
                let research_values = match values.values(device)? {
                    Diagnostic::Available(values) => {
                        account.research_values(device, values, &weights)?
                    }
                    Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                    _ => Diagnostic::UnresolvedAccuracy,
                };
                let plan_value = match values.values(device)? {
                    Diagnostic::Available(values) => {
                        let ordinary = values.dot(device, &final_demand, self.scale)?;
                        match (&self.research, &research_values) {
                            (Some(research), Diagnostic::Available(costs)) => {
                                Diagnostic::Available(ordinary.add(
                                    device,
                                    &costs.dot(
                                        device,
                                        &DeviceVector::from_ticks(device, &research.attempts)?,
                                        self.scale,
                                    )?,
                                )?)
                            }
                            (None, _) => Diagnostic::Available(ordinary),
                            (_, Diagnostic::Inadmissible) => Diagnostic::Inadmissible,
                            _ => Diagnostic::UnresolvedAccuracy,
                        }
                    }
                    Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                    _ => Diagnostic::UnresolvedAccuracy,
                };
                let coupled = match &self.reproduction {
                    Some(reproduction) => account.coupled_values(
                        device,
                        self.scale,
                        &reproduction.certificate,
                        self.budget,
                    )?,
                    None => Diagnostic::Inapplicable,
                };
                let normalized = match &coupled {
                    Diagnostic::Available(solution) => match solution.values(device)? {
                        Diagnostic::Available(values) => values.normalize(
                            device,
                            self.goods.rows() + self.reference_occupation,
                            self.scale,
                        )?,
                        Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                        _ => Diagnostic::UnresolvedAccuracy,
                    },
                    Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                    Diagnostic::Inapplicable => Diagnostic::Inapplicable,
                    _ => Diagnostic::UnresolvedAccuracy,
                };
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
                let residual_scale = match &coupled {
                    Diagnostic::Available(solution) => match solution.values(device)? {
                        Diagnostic::Available(values) => values.reciprocal_reference(
                            device,
                            self.goods.rows() + self.reference_occupation,
                            self.scale,
                        )?,
                        Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                        _ => Diagnostic::UnresolvedAccuracy,
                    },
                    Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                    Diagnostic::Inapplicable => Diagnostic::Inapplicable,
                    _ => Diagnostic::UnresolvedAccuracy,
                };
                Ok(AnalysisReport {
                    context: self.context,
                    profile: ANALYTICAL_PROFILE,
                    scale: self.scale,
                    goods_values: values.read(device)?,
                    gross_requirements: gross.read(device)?,
                    occupational_hours: hours,
                    research_values: read_diagnostic(device, research_values)?,
                    plan_value: read_diagnostic(device, plan_value)?,
                    coupled_values: match coupled {
                        Diagnostic::Available(values) => values.read(device)?,
                        Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                        Diagnostic::Inapplicable => Diagnostic::Inapplicable,
                        _ => Diagnostic::UnresolvedAccuracy,
                    },
                    normalized_coupled: read_diagnostic(device, normalized)?,
                    residual_scale: read_diagnostic(device, residual_scale)?,
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

fn read_diagnostic(
    device: &Device,
    value: Diagnostic<DeviceVector>,
) -> Result<Diagnostic<Vec<Interval>>> {
    Ok(match value {
        Diagnostic::Available(value) => Diagnostic::Available(value.read(device)?),
        Diagnostic::MissingData => Diagnostic::MissingData,
        Diagnostic::Inapplicable => Diagnostic::Inapplicable,
        Diagnostic::Inadmissible => Diagnostic::Inadmissible,
        Diagnostic::UnresolvedAccuracy => Diagnostic::UnresolvedAccuracy,
    })
}
