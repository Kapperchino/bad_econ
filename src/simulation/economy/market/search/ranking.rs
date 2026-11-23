use super::*;

#[derive(CubeType, Clone, Copy)]
pub(super) struct CandidateRank {
    pub active: u32,
    pub score: Ticks,
}

#[derive(CubeType)]
struct CandidateTree {
    scores: Shared<[u64]>,
    lanes: Shared<[u32]>,
}

#[cube]
impl workgroup::TreeStorage for CandidateTree {
    fn merge(&mut self, destination: usize, source: usize) {
        let preferred = self.scores[source] < self.scores[destination]
            || (self.scores[source] == self.scores[destination]
                && self.lanes[source] < self.lanes[destination]);
        if preferred {
            self.scores[destination] = self.scores[source];
            self.lanes[destination] = self.lanes[source];
        }
    }
}

#[cube]
pub(super) fn group_winner(rank: CandidateRank, #[comptime] mode: parallel::ScanMode) -> u32 {
    let mut winner = u32::new(comptime!(i64::from(NO_ACTOR)));
    if comptime!(mode == parallel::ScanMode::Plane) {
        let local = plane_winner(rank);
        if CUBE_DIM <= PLANE_DIM {
            winner = local.lane;
        } else {
            let mut scores = WorkgroupTree::planes::<u64>();
            let mut lanes = WorkgroupTree::planes::<u32>();
            if UNIT_POS_PLANE == 0 {
                scores[PLANE_POS as usize] = local.score;
                lanes[PLANE_POS as usize] = local.lane;
            }
            sync_cube();
            let mut best = u64::new(-1);
            for plane in 0..ceildiv(CUBE_DIM, PLANE_DIM) {
                let score = scores[plane as usize];
                let lane = lanes[plane as usize];
                if score < best || score == best && lane < winner {
                    best = score;
                    winner = lane;
                }
            }
            sync_cube();
        }
    } else {
        let score = native(rank.score);
        let mut tree = CandidateTree {
            scores: WorkgroupTree::lanes::<u64>(select(rank.active > 0, score, u64::MAX)),
            lanes: WorkgroupTree::lanes::<u32>(select(rank.active > 0, UNIT_POS, NO_ACTOR)),
        };
        WorkgroupTree::reduce::<CandidateTree>(&mut tree);
        winner = tree.lanes[0usize];
        sync_cube();
    }
    winner
}

#[derive(CubeType)]
pub(super) struct RankedLane {
    pub lane: u32,
    pub score: u64,
}

#[cube]
pub(super) fn plane_winner(rank: CandidateRank) -> RankedLane {
    let high = select(rank.active > 0, rank.score.extract(HIGH_WORD), u32::MAX);
    let minimum_high = plane_min(high);
    let low = select(
        rank.active > 0 && high == minimum_high,
        rank.score.extract(LOW_WORD),
        u32::MAX,
    );
    let minimum_low = plane_min(low);
    RankedLane {
        lane: plane_min(select(
            rank.active > 0 && high == minimum_high && low == minimum_low,
            UNIT_POS,
            NO_ACTOR,
        )),
        score: u64::cast_from(minimum_low) | (u64::cast_from(minimum_high) << 32),
    }
}

#[derive(CubeType)]
pub(super) struct MatchSelection {
    pub proposal: MarketProposal,
    pub score: Ticks,
    pub fallback: MarketProposal,
    pub fallback_score: Ticks,
    pub considered: u32,
    pub limit: u32,
    pub seen_supply: bool,
}

#[cube]
impl MatchSelection {
    pub(super) fn new(#[comptime] market: MarketFlag, limit: u32) -> Self {
        let mut score = ticks(u32::MAX, u32::MAX);
        if comptime!(market == MarketFlag::Labor) {
            score = zero();
        }
        MatchSelection {
            proposal: MarketProposal {
                peer: NO_ACTOR,
                quantity: zero(),
                trip: 0,
            },
            score,
            fallback: MarketProposal {
                peer: NO_ACTOR,
                quantity: zero(),
                trip: 0,
            },
            fallback_score: ticks(u32::MAX, u32::MAX),
            considered: 0,
            limit,
            seen_supply: false,
        }
    }

    fn choose(&mut self, candidates: &CandidateTile, slot: usize) {
        self.proposal.peer = candidates.firm[slot];
        self.proposal.quantity = candidates.quantity[slot];
        self.proposal.trip = candidates.trip[slot];
        self.score = candidates.score[slot];
    }

    fn remember_empty(&mut self, candidates: &CandidateTile, slot: usize) {
        self.fallback.peer = candidates.firm[slot];
        self.fallback.quantity = candidates.quantity[slot];
        self.fallback.trip = candidates.trip[slot];
        self.fallback_score = candidates.score[slot];
    }

    pub(super) fn rank_indexed(
        &mut self,
        candidates: &CandidateTile,
        offer: &Candidate,
        parameters: &EconomyParameters,
        window: &CandidateWindow,
    ) {
        let eligible = u32::cast_from(offer.eligible);
        let score = offer.score;
        let counted = eligible
            * u32::cast_from(
                comptime!(!matches!(parameters.market, MarketFlag::Input(_)))
                    || score.extract(HIGH_WORD) == 0,
            );
        let improving =
            eligible * u32::cast_from(self.proposal.peer == NO_ACTOR || lt(score, self.score));
        let prefix = parallel::scan_prefix(counted | (improving << 16), comptime!(parameters.scan));
        let remaining = self.limit - self.considered;
        let included = u32::cast_from(
            window.includes_lane() && (prefix.inclusive & 0xffff) - counted < remaining,
        );
        if comptime!(parameters.market == MarketFlag::Household) {
            self.seen_supply = self.seen_supply
                || parallel::reduce::<u32>(
                    u32::cast_from(offer.seen_supply) * included,
                    comptime!(Reduction::Sum),
                ) > 0;
        }
        if prefix.total >> 16 > 0 {
            let winner = group_winner(
                CandidateRank {
                    active: improving * included,
                    score,
                },
                comptime!(parameters.scan),
            );
            if winner != NO_ACTOR {
                let slot = winner as usize;
                self.choose(candidates, slot);
            }
        }
        if self.proposal.peer == NO_ACTOR {
            let fallback = group_winner(
                CandidateRank {
                    active: u32::cast_from(offer.empty) * included,
                    score,
                },
                comptime!(parameters.scan),
            );
            if fallback != NO_ACTOR {
                let slot = fallback as usize;
                let score = candidates.score[slot];
                if lt(score, self.fallback_score) {
                    self.remember_empty(candidates, slot);
                }
            }
        }
        self.considered += min(prefix.total & 0xffff, remaining);
    }

    pub(super) fn rank_plane(
        &mut self,
        candidates: &CandidateTile,
        offer: &Candidate,
        parameters: &EconomyParameters,
        control: &[Atomic<u32>],
        window: &CandidateWindow,
    ) {
        let eligible = u32::cast_from(offer.eligible);
        let counted = eligible
            * u32::cast_from(
                comptime!(!matches!(parameters.market, MarketFlag::Input(_)))
                    || offer.score.extract(HIGH_WORD) == 0,
            );
        let preceding = plane_exclusive_sum(counted);
        let visited = window.includes_lane() && preceding < self.limit - self.considered;
        let included = u32::cast_from(visited);
        let total = plane_sum(counted * included);
        self.seen_supply =
            self.seen_supply || plane_sum(u32::cast_from(offer.seen_supply) * included) > 0;
        if comptime!(parameters.market == MarketFlag::Labor) {
            let faults = candidates.faults(UNIT_POS as usize);
            if visited && faults != 0 {
                control[crate::simulation::storage::flags::FAULT_WORD].fetch_or(faults);
            }
        }
        let score = offer.score;
        let mut order = score;
        if comptime!(parameters.market == MarketFlag::Labor) {
            order = ticks(!score.extract(LOW_WORD), !score.extract(HIGH_WORD));
        }
        let winner = plane_winner(CandidateRank {
            active: eligible * included,
            score: order,
        })
        .lane;
        let fallback = plane_winner(CandidateRank {
            active: u32::cast_from(offer.empty) * included,
            score,
        })
        .lane;
        if winner != NO_ACTOR {
            let slot = winner as usize;
            let score = candidates.score[slot];
            let mut improves = lt(score, self.score);
            if comptime!(parameters.market == MarketFlag::Labor) {
                improves = lt(self.score, score);
            }
            if improves || self.proposal.peer == NO_ACTOR {
                self.choose(candidates, slot);
            }
        }
        if fallback != NO_ACTOR {
            let slot = fallback as usize;
            let score = candidates.score[slot];
            if lt(score, self.fallback_score) {
                self.remember_empty(candidates, slot);
            }
        }
        self.considered += total;
    }

    pub(super) fn rank_serial(
        &mut self,
        candidates: &mut CandidateTile,
        parameters: &EconomyParameters,
        control: &[Atomic<u32>],
        window: &CandidateWindow,
    ) {
        if UNIT_POS == 0 {
            let mut slot = 0usize;
            while slot < window.slots() && self.considered < self.limit {
                if comptime!(parameters.market == MarketFlag::Labor) {
                    let faults = candidates.faults(slot);
                    if faults != 0 {
                        control[crate::simulation::storage::flags::FAULT_WORD].fetch_or(faults);
                    }
                }
                self.seen_supply = self.seen_supply || candidates.seen_supply[slot] != 0;
                let score = candidates.score[slot];
                if candidates.empty[slot] != 0 && lt(score, self.fallback_score) {
                    self.remember_empty(&*candidates, slot);
                }
                let eligible = candidates.eligible[slot] != 0;
                let mut improves = lt(score, self.score);
                if comptime!(parameters.market == MarketFlag::Labor) {
                    improves = lt(self.score, score);
                }
                if eligible && (improves || self.proposal.peer == NO_ACTOR) {
                    self.choose(&*candidates, slot);
                }
                self.considered += u32::cast_from(
                    eligible
                        && (comptime!(!matches!(parameters.market, MarketFlag::Input(_)))
                            || score.extract(HIGH_WORD) == 0),
                );
                slot += 1;
            }
            candidates.publish_search(self.considered, u32::cast_from(self.seen_supply));
        }
        sync_cube();
        self.considered = candidates.considered();
        self.seen_supply = candidates.saw_supply();
    }
}
