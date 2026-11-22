use super::*;

#[derive(CubeType)]
pub(super) struct BuyerDemand {
    pub wanted: Ticks,
    pub purchasing_power: Ticks,
}

#[cube]
impl BuyerDemand {
    pub fn load(requests: &[u32], count: u32, buyer: u32) -> Self {
        BuyerDemand {
            wanted: ticks(
                requests[(count + buyer) as usize],
                requests[(2 * count + buyer) as usize],
            ),
            purchasing_power: ticks(
                requests[(3 * count + buyer) as usize],
                requests[(4 * count + buyer) as usize],
            ),
        }
    }

    pub fn store(&self, requests: &mut [u32], count: u32, buyer: u32) {
        requests[(count + buyer) as usize] = self.wanted.extract(LOW_WORD);
        requests[(2 * count + buyer) as usize] = self.wanted.extract(HIGH_WORD);
        requests[(3 * count + buyer) as usize] = self.purchasing_power.extract(LOW_WORD);
        requests[(4 * count + buyer) as usize] = self.purchasing_power.extract(HIGH_WORD);
    }
}
