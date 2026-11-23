use crate::domain::*;
use crate::simulation::{
    Simulation, decode,
    economy::EconomyStage,
    storage::layout::{TICK_BYTES, TICK_WORDS},
};
use anyhow::Result;

impl Simulation {
    pub fn property_snapshot(
        &mut self,
    ) -> Result<Option<crate::domain::property::PropertySnapshot>> {
        match self.scenario.config().property.is_some() {
            false => Ok(None),
            true => {
                let state = self.download()?;
                crate::domain::property::PropertySnapshot::from_state(&self.scenario, &state)
            }
        }
    }

    pub fn inspect(&mut self, request: InspectionRequest) -> Result<Inspection> {
        let index = request.row(&self.metadata.actors)?;
        let parameters = self.parameters(Phase::Audit, Market::Labor, index, 1)?;
        self.economy(
            EconomyStage::Inspect,
            self.scenario.fields(),
            &parameters,
            &self.summary,
        );
        let values = decode(&self.device.read_range(
            &self.summary,
            0,
            self.scenario.fields() * TICK_WORDS,
        )?);
        self.readback_bytes += (values.len() * TICK_BYTES) as u64;
        Inspection::from_fields(
            &self.scenario,
            &self.metadata,
            self.day,
            self.metadata.actors.at(index),
            |field| values[field as usize],
        )
    }
}
