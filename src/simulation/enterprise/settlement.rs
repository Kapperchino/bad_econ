use crate::{
    arithmetic::{add, mul},
    domain::{
        BASE_FIELDS, Field, Good, Scenario, State, TRANSIT_FIELDS,
        enterprise::{EnterpriseState, EstateStatus, Ownership, OwnershipShare},
        finance::{LoanField, LoanStatus},
        railway::CART_BILLABLE_MINUTES_PER_UNIT_DAY,
    },
};
use anyhow::{Context, Result};
use std::{
    cell::OnceCell,
    collections::{BTreeMap, HashMap},
};

pub(super) struct Settlement<'a> {
    scenario: &'a Scenario,
    heirs: HashMap<u64, usize>,
    sites: OnceCell<SettlementSites>,
}

enum Claims {
    Cleared,
    Outstanding,
}

impl Claims {
    fn new(scenario: &Scenario, state: &State, actor: usize) -> Result<Self> {
        let identity = state.actors.at(actor).identity;
        let settled = match &scenario.config().finance {
            None => true,
            Some(finance) => {
                let layout = finance.layout(state.actors.len() * scenario.fields())?;
                finance
                    .loans
                    .iter()
                    .enumerate()
                    .filter(|(_, loan)| loan.borrower == identity)
                    .all(|(index, _)| {
                        state.values[layout.loan(LoanField::Status, index)]
                            == LoanStatus::Repaid as u64
                    })
            }
        };
        Ok(match settled {
            true => Self::Cleared,
            false => Self::Outstanding,
        })
    }
}

impl<'a> Settlement<'a> {
    pub fn new(scenario: &'a Scenario, state: &State) -> Self {
        let mut heirs = HashMap::new();
        for actor in 0..state.actors.len() {
            let parent = state.get(Field::ParentHousehold, actor);
            match (
                state.actors.at(actor).kind,
                parent,
                state.get(Field::LivingMembers, actor),
            ) {
                (0, 1.., 1..) => {
                    heirs.entry(parent).or_insert(actor);
                }
                _ => (),
            }
        }
        Self {
            scenario,
            heirs,
            sites: OnceCell::new(),
        }
    }

    pub fn close_firm(&self, state: &mut State, actor: usize) -> Result<()> {
        match Claims::new(self.scenario, state, actor)? {
            Claims::Outstanding => Ok(()),
            Claims::Cleared => {
                let sites = self.sites.get_or_init(|| SettlementSites::new(state));
                liquidate(state, sites, self.scenario, actor)?;
                state.set(
                    Field::EnterpriseStatus,
                    actor,
                    EnterpriseState::Closed as u64,
                );
                Ok(())
            }
        }
    }

    pub fn inherit(&self, state: &mut State, actor: usize) -> Result<()> {
        let identity = state.actors.at(actor).identity;
        let heir = self.heirs.get(&identity).copied();
        let status = match heir {
            Some(_) => EstateStatus::AwaitingSettlement,
            None => EstateStatus::Unclaimed,
        };
        state.set(Field::EstateSettled, actor, status as u64);
        match (Claims::new(self.scenario, state, actor)?, heir) {
            (Claims::Cleared, Some(receiver)) => {
                transfer_estate(state, self.scenario, actor, receiver)?;
                state.set(
                    Field::EstateSettled,
                    actor,
                    EstateStatus::Transferred as u64,
                );
                state.set(Field::InvestmentFirm, actor, 0);
                let successor = state.actors.at(receiver).identity;
                for ownership in &mut state.ownership {
                    let mut combined = BTreeMap::<u64, u32>::new();
                    for share in &ownership.shares {
                        let household = match share.household == identity {
                            true => successor,
                            false => share.household,
                        };
                        *combined.entry(household).or_default() += share.units;
                    }
                    ownership.shares = combined
                        .into_iter()
                        .map(|(household, units)| OwnershipShare { household, units })
                        .collect();
                }
                state
                    .actors
                    .transfer_firm_ownership(actor as u32, receiver as u32);
                Ok(())
            }
            _ => Ok(()),
        }
    }

    pub fn invest(&self, state: &mut State, investor: usize, firm: usize) -> Result<()> {
        let sites = self.sites.get_or_init(|| SettlementSites::new(state));
        invest(state, sites, investor, firm)
    }
}

fn transfer_estate(state: &mut State, scenario: &Scenario, from: usize, to: usize) -> Result<()> {
    let shared_region = match &scenario.config().property {
        None => true,
        Some(property) => {
            property.region(state.actors.at(from).position)?
                == property.region(state.actors.at(to).position)?
        }
    };
    let fields = std::iter::once(Field::Cash as usize)
        .chain(Good::ALL.map(|good| Field::inventory(good) as usize))
        .chain([
            Field::Transit as usize,
            Field::RoadFreight as usize,
            Field::CartagePayable as usize,
            Field::LandPlots as usize,
            Field::CropWork as usize,
            Field::InstalledCapital as usize,
        ])
        .chain(TRANSIT_FIELDS..TRANSIT_FIELDS + Good::COUNT)
        .chain(BASE_FIELDS..BASE_FIELDS + 2 * scenario.ring() * Good::COUNT)
        .filter(|&field| {
            shared_region || ![Field::LandPlots as usize, Field::CropWork as usize].contains(&field)
        });
    let count = state.actors.len();
    for field in fields {
        let amount = state.values[field * count + from];
        state.values[field * count + to] = add(state.values[field * count + to], amount)?;
        state.values[field * count + from] = 0;
    }
    if !shared_region {
        state.set(Field::LandPlots, from, 0);
        state.set(Field::CropWork, from, 0);
    }
    Ok(())
}

struct SettlementSites {
    rows: HashMap<u64, usize>,
    issued: HashMap<u64, usize>,
    companies: HashMap<u64, usize>,
}

impl SettlementSites {
    fn new(state: &State) -> Self {
        Self {
            rows: state
                .actors
                .identities()
                .iter()
                .enumerate()
                .map(|(row, &id)| (id, row))
                .collect(),
            issued: state
                .ownership
                .iter()
                .enumerate()
                .map(|(row, issued)| (issued.firm, row))
                .collect(),
            companies: state
                .companies
                .iter()
                .enumerate()
                .flat_map(|(row, company)| company.sites.iter().map(move |&id| (id, row)))
                .collect(),
        }
    }
}

fn liquidate(
    state: &mut State,
    sites: &SettlementSites,
    scenario: &Scenario,
    firm: usize,
) -> Result<()> {
    let issued = sites
        .issued
        .get(&state.actors.at(firm).identity)
        .context("liquidation requires ownership")?;
    let ownership = state.ownership[*issued].clone();
    let mut remaining = 1_000_000u64;
    for share in ownership.shares {
        let receiver = *sites
            .rows
            .get(&share.household)
            .context("liquidation shareholder is missing")?;
        let count = state.actors.len();
        let fields = std::iter::once(Field::Cash as usize)
            .chain(Good::ALL.map(|good| Field::inventory(good) as usize))
            .chain([
                Field::LandPlots as usize,
                Field::CropWork as usize,
                Field::InstalledCapital as usize,
            ])
            .chain(BASE_FIELDS..BASE_FIELDS + 2 * scenario.ring() * Good::COUNT);
        for field in fields {
            let balance = state.values[field * count + firm];
            let amount =
                (u128::from(balance) * u128::from(share.units) / u128::from(remaining)) as u64;
            state.values[field * count + firm] = balance - amount;
            state.values[field * count + receiver] =
                add(state.values[field * count + receiver], amount)?;
        }
        for actor in [firm, receiver] {
            let mut awaiting = 0;
            let mut payable = 0;
            let mut total = 0;
            for good in Good::ALL {
                let mut transit = 0;
                for slot in 0..scenario.ring() {
                    let coordinate = state
                        .shipment_coordinate(good, slot, scenario.ring(), actor)
                        .0 as usize;
                    let waiting = state.values[coordinate + Good::COUNT * scenario.ring() * count];
                    awaiting = add(awaiting, waiting)?;
                    let rate = slot as u64
                        * u64::from(CART_BILLABLE_MINUTES_PER_UNIT_DAY)
                        * u64::from(scenario.config().wage);
                    payable = add(payable, mul(waiting, rate)?)?;
                    transit = add(transit, add(state.values[coordinate], waiting)?)?;
                }
                state.values[(TRANSIT_FIELDS + good as usize - 1) * count + actor] = transit;
                total = add(total, transit)?;
            }
            state.set(Field::Transit, actor, total);
            state.set(Field::RoadFreight, actor, awaiting);
            state.set(Field::CartagePayable, actor, payable);
        }
        remaining -= u64::from(share.units);
    }
    Ok(())
}

fn invest(state: &mut State, index: &SettlementSites, investor: usize, firm: usize) -> Result<()> {
    let paid = 5000u64;
    let household_cash = state
        .get(Field::Cash, investor)
        .checked_sub(paid)
        .context("unfunded household investment")?;
    let funds = add(state.get(Field::Cash, firm), paid)?;
    let identity = state.actors.at(firm).identity;
    let sites = index
        .companies
        .get(&identity)
        .map_or(std::slice::from_ref(&identity), |&row| {
            state.companies[row].sites.as_slice()
        });
    let book = u128::from(paid)
        + sites
            .iter()
            .filter_map(|id| index.rows.get(id))
            .map(|&site| {
                let actor = state.actors.at(site);
                u128::from(state.get(Field::Cash, site))
                    + u128::from(state.get(Field::InstalledCapital, site)) * 100
                    + u128::from(state.values[actor.kind as usize * state.actors.len() + site])
                        * u128::from(state.get(Field::Price, site))
            })
            .sum::<u128>();
    let units = (u128::from(paid) * 1_000_000 / book.max(1)).clamp(1, 250_000) as u32;
    let owner = state.actors.at(investor).identity;
    let issued = *index
        .issued
        .get(&identity)
        .context("investment requires issued shares")?;
    let mut combined = BTreeMap::<u64, u32>::new();
    for share in &state.ownership[issued].shares {
        let retained = (u64::from(share.units) * u64::from(1_000_000 - units) / 1_000_000) as u32;
        if retained > 0 {
            combined.insert(share.household, retained);
        }
    }
    *combined.entry(owner).or_default() += units;
    let allocated = combined.values().sum::<u32>();
    let founder = state
        .actors
        .at(state.actors.at(firm).owner as usize)
        .identity;
    *combined.entry(founder).or_default() += 1_000_000 - allocated;
    if combined.get(&founder).copied().unwrap_or(0) == 0 {
        combined.insert(founder, 1);
        *combined
            .get_mut(&owner)
            .context("investment shares are missing")? -= 1;
    }
    let ownership = Ownership::new(Ownership {
        firm: state.actors.at(firm).identity,
        shares: combined
            .into_iter()
            .filter(|(_, units)| *units > 0)
            .map(|(household, units)| OwnershipShare { household, units })
            .collect(),
    })?;
    for row in sites.iter().filter_map(|site| index.issued.get(site)) {
        state.ownership[*row].shares.clone_from(&ownership.shares);
    }
    state.set(Field::Cash, investor, household_cash);
    state.set(Field::Cash, firm, funds);
    state.set(Field::EquityInvested, investor, paid);
    Ok(())
}
