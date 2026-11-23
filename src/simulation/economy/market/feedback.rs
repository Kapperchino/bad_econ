use crate::domain::enterprise::FoodAccess;
use crate::simulation::economy::*;

#[allow(
    clippy::collapsible_if,
    reason = "Guard the storage read before indexing a partial workgroup"
)]
#[cube]
pub(in crate::simulation::economy) fn record_feedback(
    ledger: &mut Ledger<'_>,
    requests: &[u32],
    proposals: &mut [u32],
    outcomes: &mut [u32],
    feedback: &[Atomic<u32>],
    parameters: &EconomyParameters,
    id: u32,
) {
    let count = match comptime!(parameters.discovery) {
        Discovery::Indexed => feedback[orders::FEEDBACK_EVENTS_WORD].load(),
        #[cfg(test)]
        _ => parameters.count,
    };
    if id < count && feedback[CONSTRUCTED_EVENTS_WORD].load() == 0 {
        let index = match comptime!(parameters.discovery) {
            Discovery::Indexed => {
                let source = feedback[orders::FEEDBACK_SOURCE_WORD].load();
                if source == comptime!(orders::FeedbackSource::Searches as u32) {
                    requests[id as usize]
                } else {
                    outcomes[(parameters.sort_source + parameters.count + id) as usize]
                }
            }
            #[cfg(test)]
            _ => id,
        };
        let actor = parameters.offset + index;
        let firm = proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
        let status = KernelEventStatus::load(outcomes, parameters.status_start + index);
        if firm != NO_ACTOR && status.contains(comptime!(EventStatus::Rejected)) {
            if comptime!(parameters.market == MarketFlag::Household)
                && ledger.number(Field::FoodAccess, actor)
                    == comptime!(FoodAccess::Satisfied as u32)
            {
                ledger.set_number(
                    Field::FoodAccess,
                    actor,
                    comptime!(FoodAccess::MatchingFailure as u32),
                );
            }
            let base = CONTROL_WORDS as u32 + firm * TICK_WORDS as u32;
            let quantity =
                proposals[actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_LOW];
            let old = feedback[base as usize].fetch_add(quantity);
            let carry = u32::cast_from(quantity > u32::MAX - old);
            let high = proposals
                [actor as usize * ProposalLayout::WORDS + ProposalLayout::QUANTITY_HIGH]
                + carry;
            let prior = feedback[(base + 1) as usize].fetch_add(high);
            if high > u32::MAX - prior {
                KernelFaults::new(comptime!(FaultFlags::DEMAND_OVERFLOW)).record(feedback);
            }
        }
    }
}
#[cube]
pub(in crate::simulation::economy) fn apply_feedback(
    ledger: &mut Ledger<'_>,
    feedback: &[Atomic<u32>],
    id: u32,
) {
    let config = ledger.config;
    if id < config.actors {
        let base = CONTROL_WORDS as u32 + id * TICK_WORDS as u32;
        ledger.set(
            Field::Unfilled,
            id,
            ticks(
                feedback[base as usize].load(),
                feedback[(base + 1) as usize].load(),
            ),
        );
    }
}
