use crate::{
    arithmetic::add,
    domain::{
        Field, Good, Scenario, State,
        company::{AcquisitionScope, BusinessLine, Company, CompanySize, OwnerStubbornness},
        enterprise::{EnterpriseState, Ownership},
    },
};
use anyhow::Result;
use std::collections::{BTreeMap, BTreeSet};

pub(in crate::simulation) const PURCHASE: u32 = 11;

struct TargetValue {
    assets: i128,
    profit: i128,
}

impl TargetValue {
    fn new(state: &State, sites: &[usize]) -> Self {
        let mut value = Self {
            assets: 0,
            profit: 0,
        };
        for &site in sites {
            let actor = state.actors.at(site);
            let inventory = state.values[actor.kind as usize * state.actors.len() + site];
            let capacity = match state.get(Field::EnterpriseStatus, site) {
                status if status == EnterpriseState::Closed as u64 => 0,
                _ => i128::from(actor.capacity),
            };
            value.assets += i128::from(state.get(Field::Cash, site))
                + i128::from(inventory) * i128::from(state.get(Field::Price, site))
                + 100 * (capacity + i128::from(state.get(Field::InstalledCapital, site)));
            value.profit += i128::from(state.get(Field::BusinessClosingCash, site))
                - i128::from(state.get(Field::BusinessOpeningCash, site));
        }
        value
    }

    fn agreed_price(&self, available: u64, stubbornness: OwnerStubbornness) -> Option<u64> {
        let assets = self.assets.max(1);
        let bid = i128::from(available).min(assets + self.profit.max(0) * 7);
        let ask = (assets * i128::from(1500 + stubbornness.per_mille()) / 2000 + self.profit * 30)
            .max(assets / 2)
            .max(1);
        (bid >= ask).then_some(bid as u64)
    }
}

#[derive(Default)]
struct Holdings {
    sites: Vec<usize>,
    size: CompanySize,
    lines: BTreeSet<BusinessLine>,
}

impl Holdings {
    fn add(&mut self, state: &State, site: usize) {
        self.sites.push(site);
        if state.get(Field::EnterpriseStatus, site) != EnterpriseState::Closed as u64 {
            let actor = state.actors.at(site);
            let line = BusinessLine::from_site(actor.kind, state.get(Field::EnterpriseModel, site))
                .expect("audited enterprise");
            self.size.count_site(actor.capacity);
            self.lines.insert(line);
        }
    }

    fn permits(&self, target: &Self) -> bool {
        match self.size.acquisition_scope() {
            AcquisitionScope::ExistingLines => target.lines.is_subset(&self.lines),
            AcquisitionScope::MayDiversify => true,
        }
    }
}

struct CompanyIndex {
    actors: BTreeMap<u64, usize>,
    ownership: BTreeMap<u64, usize>,
    founders: BTreeMap<u64, u64>,
    holdings: BTreeMap<u64, Holdings>,
}

impl CompanyIndex {
    fn new(state: &State) -> Self {
        let founders = state.company_founders();
        let mut holdings = BTreeMap::<_, Holdings>::new();
        for row in state.actors.firm_indices() {
            let id = state.actors.at(row).identity;
            holdings
                .entry(founders.get(&id).copied().unwrap_or(id))
                .or_default()
                .add(state, row);
        }
        Self {
            actors: state
                .actors
                .iter()
                .enumerate()
                .map(|(row, actor)| (actor.identity, row))
                .collect(),
            ownership: state
                .ownership
                .iter()
                .enumerate()
                .map(|(row, issued)| (issued.firm, row))
                .collect(),
            founders,
            holdings,
        }
    }

    fn founder(&self, state: &State, site: usize) -> u64 {
        let id = state.actors.at(site).identity;
        self.founders.get(&id).copied().unwrap_or(id)
    }

    fn issued<'a>(&self, state: &'a State, company: u64) -> &'a Ownership {
        &state.ownership[self.ownership[&company]]
    }
}

struct Purchase {
    buyer: u64,
    seller: u64,
    bidder: usize,
    manager: u32,
    price: u64,
}

impl Purchase {
    fn negotiate(
        state: &State,
        scenario: &Scenario,
        index: &CompanyIndex,
        traded: &BTreeSet<u64>,
        bidder: usize,
        target: usize,
    ) -> Option<Self> {
        let buyer = index.founder(state, bidder);
        let seller = index.founder(state, target);
        let manager = state.actors.at(index.actors[&buyer]).owner;
        let owners = &index.issued(state, seller).shares;
        let status =
            EnterpriseState::from_index(state.get(Field::EnterpriseStatus, target)).ok()?;
        let eligible = buyer != seller
            && !traded.contains(&buyer)
            && !traded.contains(&seller)
            && index.holdings[&buyer].permits(&index.holdings[&seller])
            && state.get(Field::EnterpriseStatus, bidder) == EnterpriseState::Operating as u64
            && matches!(
                status,
                EnterpriseState::Operating | EnterpriseState::Dormant
            )
            && state.get(Field::LivingMembers, manager as usize) > 0
            && state.actors.at(index.actors[&seller]).owner != manager
            && owners
                .iter()
                .all(|share| state.get(Field::LivingMembers, index.actors[&share.household]) > 0)
            && index.issued(state, buyer).shares != *owners;
        eligible.then_some(())?;
        let floor = match state.actors.at(bidder).kind {
            kind if kind == Good::Housing as u32 => 0,
            _ => scenario.config().reserve,
        };
        let reserve = state.get(Field::OperatingReserve, bidder).max(floor);
        let available = state.get(Field::Cash, bidder).saturating_sub(reserve);
        let stubbornness = owners
            .iter()
            .map(|owner| OwnerStubbornness::from_seed(scenario.config().seed, owner.household))
            .max()?;
        let price = TargetValue::new(state, &index.holdings[&seller].sites)
            .agreed_price(available, stubbornness)?;
        Some(Self {
            buyer,
            seller,
            bidder,
            manager,
            price,
        })
    }

    fn complete(
        &self,
        state: &mut State,
        index: &CompanyIndex,
        global_row: &impl Fn(usize) -> usize,
    ) -> Result<Company> {
        state.set(
            Field::Cash,
            self.bidder,
            state.get(Field::Cash, self.bidder) - self.price,
        );
        let mut paid = 0;
        let mut units = 0u64;
        for owner in index.issued(state, self.seller).shares.clone() {
            units += u64::from(owner.units);
            let cumulative = (u128::from(self.price) * u128::from(units) / 1_000_000) as u64;
            let receiver = index.actors[&owner.household];
            state.set(
                Field::Cash,
                receiver,
                add(state.get(Field::Cash, receiver), cumulative - paid)?,
            );
            paid = cumulative;
        }
        let shares = index.issued(state, self.buyer).shares.clone();
        for &site in &index.holdings[&self.seller].sites {
            let mut actor = state.actors.at(site);
            let former = actor.owner as usize;
            if state.get(Field::InvestmentFirm, former) == global_row(site) as u64 + 1 {
                state.set(Field::InvestmentFirm, former, 0);
            }
            actor.owner = self.manager;
            state.ownership[index.ownership[&actor.identity]]
                .shares
                .clone_from(&shares);
            state.actors.set(site, actor);
        }
        let mut sites = index.holdings[&self.buyer]
            .sites
            .iter()
            .chain(&index.holdings[&self.seller].sites)
            .map(|&row| state.actors.at(row).identity)
            .collect::<Vec<_>>();
        sites.sort_unstable();
        Ok(Company {
            identity: self.buyer,
            sites,
        })
    }
}

pub(super) fn settle(
    state: &mut State,
    scenario: &Scenario,
    requests: &[u32],
    global_row: impl Fn(usize) -> usize,
) -> Result<()> {
    let mut bids = requests
        .iter()
        .enumerate()
        .filter(|(_, request)| **request >> 24 == PURCHASE)
        .peekable();
    if bids.peek().is_some() {
        let index = CompanyIndex::new(state);
        let mut traded = BTreeSet::new();
        let mut merged = Vec::new();
        for (bidder, request) in bids {
            let target = (request & 0x00ff_ffff) as usize - 1;
            if let Some(sale) =
                Purchase::negotiate(state, scenario, &index, &traded, bidder, target)
            {
                merged.push(sale.complete(state, &index, &global_row)?);
                traded.extend([sale.buyer, sale.seller]);
            }
        }
        state
            .companies
            .retain(|company| !traded.contains(&company.identity));
        state.companies.extend(merged);
    }
    Ok(())
}
