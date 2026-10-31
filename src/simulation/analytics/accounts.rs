use super::*;

pub struct AccountModel {
    goods: SparseMatrix,
    labor: SparseMatrix,
    reproduction: Option<ReproductionModel>,
    research: Option<ResearchModel>,
}

pub struct ReproductionModel {
    goods: SparseMatrix,
    training: SparseMatrix,
}

pub struct ResearchModel {
    inputs: SparseMatrix,
    hours: SparseMatrix,
}

impl AccountModel {
    pub fn new(goods: SparseMatrix, labor: SparseMatrix) -> Result<Self> {
        match goods.rows == goods.columns && goods.rows == labor.columns && labor.rows > 0 {
            true => Ok(Self {
                goods,
                labor,
                reproduction: None,
                research: None,
            }),
            false => Err(anyhow!(
                "goods and occupational account dimensions disagree"
            )),
        }
    }

    pub fn with_reproduction(
        mut self,
        goods: SparseMatrix,
        training: SparseMatrix,
    ) -> Result<Self> {
        let _ = SparseMatrix::coupled(&self.goods, &self.labor, &goods, &training)?;
        self.reproduction = Some(ReproductionModel { goods, training });
        Ok(self)
    }

    pub fn with_research(mut self, inputs: SparseMatrix, hours: SparseMatrix) -> Result<Self> {
        match inputs.rows == self.goods.rows
            && hours.rows == self.labor.rows
            && inputs.columns == hours.columns
        {
            true => {
                self.research = Some(ResearchModel { inputs, hours });
                Ok(self)
            }
            false => Err(anyhow!(
                "research input and occupational-hour dimensions disagree"
            )),
        }
    }

    pub fn upload(&self, device: &Device) -> Result<DeviceAccounts> {
        let coupled = self
            .reproduction
            .as_ref()
            .map(|reproduction| -> Result<_> {
                Ok(DeviceOperator::new(
                    device,
                    &SparseMatrix::coupled(
                        &self.goods,
                        &self.labor,
                        &reproduction.goods,
                        &reproduction.training,
                    )?,
                ))
            })
            .transpose()?;
        let research = self.research.as_ref().map(|research| DeviceResearch {
            inputs: DeviceOperator::new(device, &research.inputs),
            input_values: DeviceOperator::new(device, &research.inputs.transpose()),
            hours: DeviceOperator::new(device, &research.hours),
            hour_values: DeviceOperator::new(device, &research.hours.transpose()),
        });
        Ok(DeviceAccounts {
            goods: DeviceOperator::new(device, &self.goods),
            goods_values: DeviceOperator::new(device, &self.goods.transpose()),
            labor: DeviceOperator::new(device, &self.labor),
            labor_values: DeviceOperator::new(device, &self.labor.transpose()),
            coupled,
            research,
        })
    }
}

struct DeviceResearch {
    inputs: DeviceOperator,
    input_values: DeviceOperator,
    hours: DeviceOperator,
    hour_values: DeviceOperator,
}

pub struct DeviceAccounts {
    goods: DeviceOperator,
    goods_values: DeviceOperator,
    labor: DeviceOperator,
    labor_values: DeviceOperator,
    coupled: Option<DeviceOperator>,
    research: Option<DeviceResearch>,
}

pub struct PlanAccount {
    pub gross_requirements: DeviceSolution,
    pub occupational_hours: Diagnostic<DeviceVector>,
}

impl DeviceAccounts {
    pub fn external_values(
        &self,
        device: &Device,
        weights: &DeviceVector,
        certificate: &Contraction,
        budget: SolveBudget,
    ) -> Result<DeviceSolution> {
        let forcing = self.labor_values.apply(device, weights)?;
        self.goods_values
            .solve(device, &forcing, certificate, budget)
    }

    pub fn goods_requirements(
        &self,
        device: &Device,
        demand: &DeviceVector,
        certificate: &Contraction,
        budget: SolveBudget,
    ) -> Result<DeviceSolution> {
        self.goods.solve(device, demand, certificate, budget)
    }

    pub fn coupled_values(
        &self,
        device: &Device,
        scale: u32,
        certificate: &Contraction,
        budget: SolveBudget,
    ) -> Result<Diagnostic<DeviceSolution>> {
        match &self.coupled {
            None => Ok(Diagnostic::Inapplicable),
            Some(coupled) => match scale {
                0 => Err(anyhow!("analytical scale must be positive")),
                _ => {
                    let rhs = (0..coupled.rows)
                        .map(|row| match row < self.goods.rows {
                            true => 0,
                            false => u64::from(scale),
                        })
                        .collect::<Vec<_>>();
                    let solution = coupled.solve(
                        device,
                        &DeviceVector::exact(device, &rhs)?,
                        certificate,
                        budget,
                    )?;
                    Ok(match solution.values(device)? {
                        Diagnostic::Available(_) => Diagnostic::Available(solution),
                        Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                        _ => Diagnostic::UnresolvedAccuracy,
                    })
                }
            },
        }
    }

    pub fn research_values(
        &self,
        device: &Device,
        goods_values: &DeviceVector,
        weights: &DeviceVector,
    ) -> Result<Diagnostic<DeviceVector>> {
        match &self.research {
            None => Ok(Diagnostic::Inapplicable),
            Some(research) => Ok(Diagnostic::Available(
                research
                    .input_values
                    .apply(device, goods_values)?
                    .add(device, &research.hour_values.apply(device, weights)?)?,
            )),
        }
    }

    pub fn plan(
        &self,
        device: &Device,
        final_demand: &DeviceVector,
        attempts: &DeviceVector,
        certificate: &Contraction,
        budget: SolveBudget,
    ) -> Result<Diagnostic<PlanAccount>> {
        match &self.research {
            None => Ok(Diagnostic::Inapplicable),
            Some(research) => {
                let demand = final_demand.add(device, &research.inputs.apply(device, attempts)?)?;
                let gross_requirements = self.goods.solve(device, &demand, certificate, budget)?;
                let occupational_hours = match gross_requirements.values(device)? {
                    Diagnostic::Available(requirements) => Diagnostic::Available(
                        self.labor
                            .apply(device, requirements)?
                            .add(device, &research.hours.apply(device, attempts)?)?,
                    ),
                    Diagnostic::Inadmissible => Diagnostic::Inadmissible,
                    _ => Diagnostic::UnresolvedAccuracy,
                };
                Ok(Diagnostic::Available(PlanAccount {
                    gross_requirements,
                    occupational_hours,
                }))
            }
        }
    }

    pub fn plan_value(
        device: &Device,
        goods_values: &DeviceVector,
        research_values: &DeviceVector,
        final_demand: &DeviceVector,
        attempts: &DeviceVector,
        scale: u32,
    ) -> Result<DeviceVector> {
        goods_values
            .dot(device, final_demand, scale)?
            .add(device, &research_values.dot(device, attempts, scale)?)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComparatorContext {
    pub population: String,
    pub classification: String,
    pub period: String,
    pub convention: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Sign {
    NonNegative,
    Negative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SignedTicks {
    pub sign: Sign,
    pub magnitude: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct SignedInterval {
    pub lower: SignedTicks,
    pub upper: SignedTicks,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct ComparatorReport {
    pub context: ComparatorContext,
    pub surplus: SignedInterval,
    pub rate: Diagnostic<SignedInterval>,
    pub rate_scale: u32,
}

pub fn comparator(
    device: &Device,
    context: ComparatorContext,
    living: u64,
    requirement: &DeviceVector,
    scale: u32,
) -> Result<ComparatorReport> {
    let labels = [
        &context.population,
        &context.classification,
        &context.period,
        &context.convention,
    ];
    match requirement.length == 1
        && scale > 0
        && labels.iter().all(|label| !label.trim().is_empty())
    {
        false => Err(anyhow!(
            "comparator requires complete context, a scalar requirement, and a positive scale"
        )),
        true => {
            let output = device.empty(12);
            let control = device.tensor(&[0; CONTROL_WORDS]);
            kernels::comparator::launch(
                &device.client,
                CubeCount::Static(1, 1, 1),
                CubeDim::new_1d(device.workgroup),
                argument(&requirement.data),
                argument(&output),
                argument(&control),
                living as u32,
                (living >> 32) as u32,
                scale,
            );
            let status = faults(device, &control)?;
            let words = device.read(&output)?;
            let signed = |offset| SignedTicks {
                sign: match words[offset] {
                    0 => Sign::NonNegative,
                    _ => Sign::Negative,
                },
                magnitude: decode(&words[offset + 1..offset + 3]),
            };
            Ok(ComparatorReport {
                context,
                surplus: SignedInterval {
                    lower: signed(0),
                    upper: signed(3),
                },
                rate: match status[0] {
                    0 => Diagnostic::Available(SignedInterval {
                        lower: signed(6),
                        upper: signed(9),
                    }),
                    kernels::INAPPLICABLE => Diagnostic::Inapplicable,
                    _ => Diagnostic::UnresolvedAccuracy,
                },
                rate_scale: scale,
            })
        }
    }
}
