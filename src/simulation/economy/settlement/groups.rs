use super::super::*;

#[derive(CubeType)]
pub(in crate::simulation::economy) struct GroupedOrders<'a> {
    proposals: &'a mut [u32],
    scratch: &'a mut [u32],
    control: &'a [Atomic<u32>],
    parameters: &'a EconomyParameters,
}

#[cube]
impl<'a> GroupedOrders<'a> {
    pub(in crate::simulation::economy) fn new(
        proposals: &'a mut [u32],
        scratch: &'a mut [u32],
        control: &'a [Atomic<u32>],
        parameters: &'a EconomyParameters,
    ) -> Self {
        GroupedOrders::<'a> {
            proposals,
            scratch,
            control,
            parameters,
        }
    }

    pub(in crate::simulation::economy) fn dispatch(
        &mut self,
        economy: &mut EconomyView<'_>,
        #[comptime] stage: EconomyStage,
        id: u32,
    ) {
        match comptime!(stage) {
            EconomyStage::GroupInitialize => self.initialize(id),
            EconomyStage::GroupSort => self.sort(id),
            EconomyStage::GroupMerge => self.merge(id),
            EconomyStage::SettleInputTile => self.settle_input_tile(economy),
            EconomyStage::GroupExecute => self.execute(economy, id),
            EconomyStage::GroupRemainder => self.fill_remainder(economy, id),
            _ => (),
        }
    }

    fn initialize(&mut self, id: u32) {
        let control = self.control;
        let parameters = self.parameters;
        let count = parameters.count;
        let mut peer = u32::new(comptime!(i64::from(NO_ACTOR)));
        if id < count {
            peer = self.proposals
                [(parameters.offset + id) as usize * ProposalLayout::WORDS + ProposalLayout::ACTOR];
            KernelEventStatus::new(comptime!(EventStatus::Inactive))
                .store(self.scratch, 4 * count + id);
        }
        let prefix = parallel::prefix_sum::<u32>(u32::cast_from(peer != NO_ACTOR));
        let mut first = Shared::<u32>::new();
        if UNIT_POS == 0 {
            *first = control[CONSTRUCTED_EVENTS_WORD].fetch_add(prefix.total);
        }
        sync_cube();
        if peer != NO_ACTOR {
            let slot = *first + prefix.inclusive - 1;
            self.scratch[slot as usize] = peer;
            self.scratch[(count + slot) as usize] = id;
        }
    }

    fn sort(&mut self, id: u32) {
        let control = self.control;
        let parameters = self.parameters;
        let count = parameters.count;
        let active = orders::active(control, parameters);
        let lane = UNIT_POS as usize;
        let mut keys = Shared::<[u32]>::new_slice(256usize);
        let mut items = Shared::<[u32]>::new_slice(256usize);
        let mut key = u32::new(comptime!(i64::from(NO_ACTOR)));
        let mut item = 0u32;
        if id < active {
            key = self.scratch[id as usize];
            item = self.scratch[(count + id) as usize];
        }
        keys[lane] = key;
        items[lane] = item;
        sync_cube();
        let rotation = parameters.settlement_rotation % max(count, 1);
        let mut width = 2u32;
        while width <= CUBE_DIM {
            let mut stride = width / 2;
            while stride > 0 {
                let other = (UNIT_POS ^ stride) as usize;
                let other_key = keys[other];
                let other_item = items[other];
                let priority = (item + count - rotation) % max(count, 1);
                let other_priority = (other_item + count - rotation) % max(count, 1);
                let precedes = other_key < key || (other_key == key && other_priority < priority);
                let lower = (UNIT_POS & width == 0) == (UNIT_POS & stride == 0);
                if precedes == lower {
                    key = other_key;
                    item = other_item;
                }
                sync_cube();
                keys[lane] = key;
                items[lane] = item;
                sync_cube();
                stride /= 2;
            }
            width *= 2;
        }
        if id < active {
            self.scratch[id as usize] = key;
            self.scratch[(count + id) as usize] = item;
        }
    }

    fn merge(&mut self, id: u32) {
        let control = self.control;
        let parameters = self.parameters;
        let count = parameters.count;
        let active = orders::active(control, parameters);
        if id < active {
            let width = parameters.sort_width;
            let source = parameters.sort_source;
            let destination = 2 * count - source;
            let base = id / (width * 2) * width * 2;
            let middle = min(base + width, active);
            let end = min(base + width * 2, active);
            let first = id < middle;
            let start = select(first, middle, base);
            let stop = select(first, end, middle);
            let key = self.scratch[(source + id) as usize];
            let item = self.scratch[(source + count + id) as usize];
            let mut low = start;
            let mut high = stop;
            while low < high {
                let probe = (low + high) / 2;
                let other_key = self.scratch[(source + probe) as usize];
                let other_item = self.scratch[(source + count + probe) as usize];
                let rotation = parameters.settlement_rotation % max(count, 1);
                let priority = (item + count - rotation) % max(count, 1);
                let other_priority = (other_item + count - rotation) % max(count, 1);
                let precedes = other_key < key || (other_key == key && other_priority < priority);
                low = select(precedes, probe + 1, low);
                high = select(precedes, high, probe);
            }
            let target = destination + base + id - select(first, base, middle) + low - start;
            self.scratch[target as usize] = key;
            self.scratch[(target + count) as usize] = item;
        }
    }

    fn execute(&mut self, economy: &mut EconomyView<'_>, id: u32) {
        let control = self.control;
        let parameters = self.parameters;
        let range = id;
        let count = parameters.count;
        let active_count = orders::active(control, parameters);
        let mut invocation = id;
        if comptime!(parameters.discovery == Discovery::Indexed) {
            invocation = active_count;
            if id < control[RANGE_SLOTS_WORD].load() {
                invocation = self.scratch[(5 * count + id) as usize];
            }
        }
        let id = invocation;
        if id < active_count {
            let source = parameters.sort_source;
            let key = self.scratch[(source + id) as usize];
            let mut first = id == 0;
            if id > 0 {
                first = self.scratch[(source + id - 1) as usize] != key;
            }
            if first {
                let low = id;
                let mut cursor = low;
                let mut active = true;
                let mut event = Array::<u32>::new(EVENT_WORDS);
                let mut peer = PeerBalances::new(economy, parameters, key);
                let deferred = comptime!(
                    parameters.market == MarketFlag::Household
                        && parameters.discovery == Discovery::Indexed
                );
                if deferred {
                    active = peer.has_supply();
                }
                while cursor < active_count && active {
                    active = self.scratch[(source + cursor) as usize] == key;
                    if active {
                        let actor = self.scratch[(source + count + cursor) as usize];
                        if comptime!(
                            parameters.market == MarketFlag::Payroll
                                || parameters.market == MarketFlag::Distribution
                        ) {
                            construct(
                                economy,
                                self.proposals,
                                &mut event[..],
                                control,
                                parameters,
                                actor,
                                comptime!(EventStorage::Local),
                            );
                            apply(economy.ledger.words, &event[..], control, 0)
                                .store(self.scratch, 4 * count + actor);
                        } else {
                            let prior = KernelEventStatus::load(self.scratch, 4 * count + actor);
                            peer.offer(
                                &mut economy.ledger,
                                self.proposals,
                                control,
                                parameters,
                                PeerOffer {
                                    actor: actor + parameters.offset,
                                    peer: key,
                                    status: prior,
                                },
                            )
                            .store(self.scratch, 4 * count + actor);
                        }
                        cursor += 1;
                        if deferred {
                            active = peer.has_supply();
                        }
                    }
                }
                if deferred {
                    self.scratch[(7 * count + key) as usize] = cursor;
                    cursor = low + self.scratch[(6 * count + range) as usize];
                }
                control[PROGRESS_WORD].fetch_max(cursor - low);
                control[COMPLETED_WORD].fetch_add(cursor - low);
                peer.store(economy, parameters, key);
            }
        }
    }

    fn fill_remainder(&mut self, economy: &mut EconomyView<'_>, id: u32) {
        let control = self.control;
        let parameters = self.parameters;
        let count = parameters.count;
        if id < orders::active(control, parameters) {
            let key = self.scratch[(parameters.sort_source + id) as usize];
            if id >= self.scratch[(7 * count + key) as usize] {
                let actor = self.scratch[(parameters.sort_source + count + id) as usize];
                let prior = KernelEventStatus::load(self.scratch, 4 * count + actor);
                let mut peer = PeerBalances::new(economy, parameters, key);
                peer.offer(
                    &mut economy.ledger,
                    self.proposals,
                    control,
                    parameters,
                    PeerOffer {
                        actor,
                        peer: key,
                        status: prior,
                    },
                )
                .store(self.scratch, 4 * count + actor);
            }
        }
    }

    fn settle_input_tile(&mut self, economy: &mut EconomyView<'_>) {
        let orders = orders::active(self.control, self.parameters);
        if self.control[LATCH_WORD].load() == 0 {
            let mut slot = UNIT_POS;
            while slot < orders {
                actor_settlement(
                    economy,
                    self.proposals,
                    self.scratch,
                    self.control,
                    self.parameters,
                    slot,
                    comptime!(Posting::CheckActor),
                );
                slot += CUBE_DIM;
            }
            sync_storage();
            let mut slot = UNIT_POS;
            while slot < orders {
                self.execute(economy, slot);
                slot += CUBE_DIM;
            }
            sync_storage();
            if UNIT_POS == 0 {
                self.control[LATCH_WORD]
                    .store(self.control[crate::simulation::flags::FAULT_WORD].load());
            }
            sync_storage();
            if self.control[LATCH_WORD].load() == 0 {
                let mut slot = UNIT_POS;
                while slot < orders {
                    actor_settlement(
                        economy,
                        self.proposals,
                        self.scratch,
                        self.control,
                        self.parameters,
                        slot,
                        comptime!(Posting::ApplyActor),
                    );
                    slot += CUBE_DIM;
                }
            }
        }
    }
}
