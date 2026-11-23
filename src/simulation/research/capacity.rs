use super::{
    access::{ResearchView, ResearchViewExpand},
    layout::*,
};
use crate::{
    domain::{Field, research::RESEARCH_SCALE},
    simulation::storage::{
        state::{ReadLedger, ReadLedgerExpand},
        words::*,
    },
};
use cubecl::prelude::*;

#[cube]
impl<'a> ResearchView<'a> {
    pub fn open_machine(&mut self, id: usize) {
        if id < self.parameters.machine_count as usize {
            let machine = self.parameters.machine(id);
            let produced = self.main.value(
                Field::UnscaledProduced,
                self.records
                    .number(&machine.definition, MachineLayout::FIRM),
            );
            self.state
                .set_amount(&machine.saved, MachineState::PRODUCED_BEFORE, produced);
            self.state
                .set_number(&machine.saved, MachineState::MAINTENANCE_RESTORATION, 0);
        }
    }

    pub fn open_stock(&mut self, id: usize) {
        if id < self.parameters.stock_count as usize {
            let stock = self.parameters.stock(id);
            let produced = self.main.value(
                Field::UnscaledProduced,
                self.records.number(&stock.definition, StockLayout::FIRM),
            );
            self.state
                .set_amount(&stock.saved, StockState::PRODUCED_BEFORE, produced);
        }
    }

    pub fn machine_capacity(&mut self, id: usize) {
        if id < self.parameters.machine_count as usize {
            let machine = self.parameters.machine(id);
            let nominal = self
                .records
                .number(&machine.definition, MachineLayout::NAMEPLATE)
                - self
                    .records
                    .number(&machine.definition, MachineLayout::DOWNTIME);
            let capacity = minimum(
                self.state.amount(&machine.saved, MachineState::REMAINING),
                ticks(
                    self.state.number(&machine.saved, MachineState::CONDITION) * nominal
                        / RESEARCH_SCALE,
                    0,
                ),
            );
            self.cap_plan(
                self.records
                    .number(&machine.definition, MachineLayout::FIRM),
                capacity,
            );
        }
    }

    pub fn stock_capacity(&mut self, id: usize) {
        if id < self.parameters.stock_count as usize {
            let stock = self.parameters.stock(id);
            self.cap_plan(
                self.records.number(&stock.definition, StockLayout::FIRM),
                ticks(
                    self.state.number(&stock.saved, StockState::STOCK)
                        / self
                            .records
                            .number(&stock.definition, StockLayout::EXTRACTION),
                    0,
                ),
            );
        }
    }

    pub fn maintain(&mut self, id: usize) {
        if id < self.parameters.machine_count as usize {
            let machine = self.parameters.machine(id);
            let good = self
                .records
                .number(&machine.definition, MachineLayout::MAINTENANCE_GOOD);
            let firm = self
                .records
                .number(&machine.definition, MachineLayout::FIRM);
            let required = ticks(
                self.records
                    .number(&machine.definition, MachineLayout::MAINTENANCE_TICKS),
                0,
            );
            let available = self.main.value_at(good, firm);
            if !lt(available, required) {
                self.main.set_at(good, firm, minus(available, required));
                self.state.increase(
                    &machine.saved,
                    MachineState::CONSUMED,
                    required,
                    self.control,
                );
                self.state.set_number(
                    &machine.saved,
                    MachineState::MAINTENANCE_RESTORATION,
                    self.records
                        .number(&machine.definition, MachineLayout::MAINTENANCE_TICKS)
                        * self
                            .records
                            .number(&machine.definition, MachineLayout::RESTORATION),
                );
            }
        }
    }

    pub fn close_machine(&mut self, id: usize) {
        if id < self.parameters.machine_count as usize {
            let machine = self.parameters.machine(id);
            let used = self.executed_packets(
                self.records
                    .number(&machine.definition, MachineLayout::FIRM),
            );
            let remaining = self.state.amount(&machine.saved, MachineState::REMAINING);
            if lt(remaining, used)
                || lt(
                    ticks(
                        self.records
                            .number(&machine.definition, MachineLayout::MAXIMUM_PACKETS),
                        0,
                    ),
                    used,
                )
            {
                self.record_mismatch();
            }
            self.state.set_amount(
                &machine.saved,
                MachineState::REMAINING,
                positive(remaining, used),
            );
            let wear = used.extract(0usize)
                * self
                    .records
                    .number(&machine.definition, MachineLayout::WEAR);
            let repaired = self.state.number(&machine.saved, MachineState::CONDITION)
                + self
                    .state
                    .number(&machine.saved, MachineState::MAINTENANCE_RESTORATION);
            self.state.set_number(
                &machine.saved,
                MachineState::CONDITION,
                min(RESEARCH_SCALE, repaired - min(repaired, wear)),
            );
            self.state
                .set_number(&machine.saved, MachineState::MAINTENANCE_RESTORATION, 0);
        }
    }

    pub fn close_stock(&mut self, id: usize) {
        if id < self.parameters.stock_count as usize {
            let stock = self.parameters.stock(id);
            let used =
                self.executed_packets(self.records.number(&stock.definition, StockLayout::FIRM));
            let extraction = times(
                used,
                self.records
                    .number(&stock.definition, StockLayout::EXTRACTION),
            );
            extraction.faults.record(self.control);
            let available = self.state.number(&stock.saved, StockState::STOCK);
            if lt(ticks(available, 0), extraction.value)
                || lt(
                    ticks(
                        self.records
                            .number(&stock.definition, StockLayout::MAXIMUM_PACKETS),
                        0,
                    ),
                    used,
                )
            {
                self.record_mismatch();
            }
            let remaining = positive(ticks(available, 0), extraction.value).extract(0usize);
            let capacity = self
                .records
                .number(&stock.definition, StockLayout::CAPACITY);
            let growth = ratio(
                ticks(remaining, 0),
                self.records.number(&stock.definition, StockLayout::RENEWAL)
                    * (capacity - remaining),
                capacity * RESEARCH_SCALE,
            );
            growth.faults.record(self.control);
            self.state.set_number(
                &stock.saved,
                StockState::STOCK,
                remaining + growth.value.extract(0usize),
            );
            self.state.increase(
                &stock.saved,
                StockState::EXTRACTED,
                extraction.value,
                self.control,
            );
        }
    }
}
