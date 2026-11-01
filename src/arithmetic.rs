use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const SCALE: u64 = 1_000;
pub const PROFILE: &str = "DeterministicV1";
pub const PRICE_TABLE: [u32; 65] = [
    951230, 952717, 954207, 955699, 957194, 958691, 960190, 961691, 963195, 964701, 966210, 967720,
    969234, 970749, 972267, 973788, 975310, 976836, 978363, 979893, 981425, 982960, 984497, 986036,
    987578, 989123, 990669, 992218, 993770, 995324, 996880, 998439, 1000000, 1001564, 1003130,
    1004699, 1006270, 1007844, 1009420, 1010998, 1012579, 1014162, 1015748, 1017337, 1018927,
    1020521, 1022117, 1023715, 1025316, 1026919, 1028525, 1030133, 1031744, 1033357, 1034973,
    1036592, 1038212, 1039836, 1041462, 1043091, 1044722, 1046355, 1047992, 1049630, 1051272,
];

pub fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b).context("resource overflow")
}
pub fn mul(a: u64, b: u64) -> Result<u64> {
    a.checked_mul(b).context("resource product overflow")
}
pub fn floor_ratio(a: u64, b: u64, divisor: u64) -> Result<u64> {
    match divisor {
        0 => Err(anyhow!("zero divisor")),
        d => u64::try_from(u128::from(a) * u128::from(b) / u128::from(d)).context("ratio overflow"),
    }
}
pub fn ceil_ratio(a: u64, b: u64, divisor: u64) -> Result<u64> {
    match divisor {
        0 => Err(anyhow!("zero divisor")),
        d => u64::try_from((u128::from(a) * u128::from(b)).div_ceil(u128::from(d)))
            .context("ratio overflow"),
    }
}
pub fn quote(candidate: u64, low: u64, high: u64, quantum: u64) -> Result<u64> {
    let lo = ceil_ratio(low, 1, quantum)?;
    let hi = floor_ratio(high, 1, quantum)?;
    match lo <= hi {
        true => mul(ceil_ratio(candidate, 1, quantum)?.clamp(lo, hi), quantum),
        false => Err(anyhow!("empty quote grid")),
    }
}
pub fn forecast(old: u64, sales: u64, unfilled: u64, weight: u64) -> Result<u64> {
    let signal = add(sales, unfilled)?;
    let numerator = u128::from(old)
        * u128::from(SCALE.checked_sub(weight).context("invalid weight")?)
        + u128::from(signal) * u128::from(weight);
    u64::try_from(numerator / u128::from(SCALE)).context("forecast overflow")
}
pub fn price(
    price: u64,
    unfilled: u64,
    excess: u64,
    demand: u64,
    low: u64,
    high: u64,
) -> Result<u64> {
    let imbalance = (i128::from(unfilled) - i128::from(excess)) * 32 / i128::from(demand.max(1));
    let index = usize::try_from(32 + imbalance.clamp(-32, 32))?;
    quote(
        ceil_ratio(price, u64::from(PRICE_TABLE[index]), 1_000_000)?,
        low,
        high,
        1,
    )
}
pub fn table_hash() -> String {
    hex(&Sha256::digest(
        PRICE_TABLE
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    ))
}
pub fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "ProbabilityDefinition", into = "ProbabilityDefinition")]
pub struct Probability {
    numerator: u64,
    denominator: u64,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProbabilityDefinition {
    numerator: u64,
    denominator: u64,
}

impl TryFrom<ProbabilityDefinition> for Probability {
    type Error = anyhow::Error;

    fn try_from(value: ProbabilityDefinition) -> Result<Self> {
        Self::new(value.numerator, value.denominator)
    }
}

impl From<Probability> for ProbabilityDefinition {
    fn from(value: Probability) -> Self {
        Self {
            numerator: value.numerator,
            denominator: value.denominator,
        }
    }
}

impl Probability {
    pub fn new(numerator: u64, denominator: u64) -> Result<Self> {
        match denominator > 0 && numerator <= denominator {
            true => Ok(Self {
                numerator,
                denominator,
            }),
            false => Err(anyhow!("probability must lie in [0,1]")),
        }
    }
    pub fn succeeds(self, bits: u32, draw: u64) -> Result<bool> {
        let range = 1_u128
            .checked_shl(bits)
            .filter(|_| bits <= 64)
            .context("at most 64 draw bits")?;
        match u128::from(draw) < range {
            true => {
                match self.numerator {
                    0 => Ok(false),
                    numerator if numerator == self.denominator => Ok(true),
                    numerator => Ok(u128::from(draw)
                        < range * u128::from(numerator) / u128::from(self.denominator)),
                }
            }
            false => Err(anyhow!("draw exceeds its declared bit width")),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct DrawKey {
    pub seed: u64,
    pub day: u64,
    pub phase: u32,
    pub actor: u64,
    pub purpose: u32,
    pub index: u64,
}
impl DrawKey {
    pub fn draw(self) -> u64 {
        let mut hash = Sha256::new();
        hash.update(b"bad-econ/draw/DeterministicV1\0");
        hash.update(self.seed.to_le_bytes());
        hash.update(self.day.to_le_bytes());
        hash.update(self.phase.to_le_bytes());
        hash.update(self.actor.to_le_bytes());
        hash.update(self.purpose.to_le_bytes());
        hash.update(self.index.to_le_bytes());
        let bytes = hash.finalize();
        u64::from_le_bytes(bytes[..8].try_into().expect("SHA-256 has 32 bytes"))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Claim {
    denominator: u64,
    numerator: u64,
}
impl Claim {
    pub fn new(denominator: u64, numerator: u64) -> Result<Self> {
        match denominator {
            0 => Err(anyhow!("zero claim denominator")),
            _ => Ok(Self {
                denominator,
                numerator,
            }),
        }
    }
    pub fn accrue(&self, numerator: u64) -> Result<Self> {
        Self::new(self.denominator, add(self.numerator, numerator)?)
    }
    pub fn whole(&self) -> u64 {
        self.numerator / self.denominator
    }
    pub fn remainder(&self) -> u64 {
        self.numerator % self.denominator
    }
    pub fn settle(&self, ticks: u64) -> Result<Self> {
        Self::new(
            self.denominator,
            self.numerator
                .checked_sub(mul(ticks, self.denominator)?)
                .context("unfunded claim")?,
        )
    }
}

#[derive(Debug, Clone)]
pub struct Packet {
    pub activities: u64,
    pub ticks: Vec<u64>,
}
impl Packet {
    pub fn new(legs: &[Ratio]) -> Result<Self> {
        let activities = legs.iter().try_fold(1, |lcm, r| {
            mul(lcm / gcd(lcm, r.denominator), r.denominator)
        })?;
        let ticks = legs
            .iter()
            .map(|r| mul(activities / r.denominator, r.numerator))
            .collect::<Result<_>>()?;
        Ok(Self { activities, ticks })
    }
    pub fn capacity(&self, plan: u64, available: &[u64]) -> Result<u64> {
        match available.len() == self.ticks.len() {
            true => Ok(available
                .iter()
                .zip(&self.ticks)
                .fold(plan, |count, (&stock, &debit)| match debit {
                    0 => count,
                    _ => count.min(stock / debit),
                })),
            false => Err(anyhow!("packet resource shape mismatch")),
        }
    }
}
#[derive(Debug, Clone)]
pub struct Ratio {
    numerator: u64,
    denominator: u64,
}
impl Ratio {
    pub fn new(numerator: u64, denominator: u64) -> Result<Self> {
        match denominator {
            0 => Err(anyhow!("zero ratio denominator")),
            _ => {
                let factor = gcd(numerator, denominator);
                Ok(Self {
                    numerator: numerator / factor,
                    denominator: denominator / factor,
                })
            }
        }
    }
}
fn gcd(a: u64, b: u64) -> u64 {
    match b {
        0 => a,
        _ => gcd(b, a % b),
    }
}
