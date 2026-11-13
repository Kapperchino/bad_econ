use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::fmt;

pub const DAYS_PER_YEAR: u32 = 365;
pub const HARVEST_START: u32 = 270;
pub const HARVEST_DAYS: u32 = 30;

#[repr(u32)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Month {
    January = 1,
    February,
    March,
    April,
    May,
    June,
    July,
    August,
    September,
    October,
    November,
    December,
}

impl Month {
    pub fn from_number(number: u32) -> Result<Self> {
        match number {
            1 => Ok(Self::January),
            2 => Ok(Self::February),
            3 => Ok(Self::March),
            4 => Ok(Self::April),
            5 => Ok(Self::May),
            6 => Ok(Self::June),
            7 => Ok(Self::July),
            8 => Ok(Self::August),
            9 => Ok(Self::September),
            10 => Ok(Self::October),
            11 => Ok(Self::November),
            12 => Ok(Self::December),
            _ => Err(anyhow!("month must be between 1 and 12")),
        }
    }

    pub fn days(self) -> u32 {
        match self {
            Self::February => 28,
            Self::April | Self::June | Self::September | Self::November => 30,
            _ => 31,
        }
    }

    fn first_day(self) -> u32 {
        [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334][self as usize - 1]
    }
}

impl fmt::Display for Month {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::January => "Jan",
            Self::February => "Feb",
            Self::March => "Mar",
            Self::April => "Apr",
            Self::May => "May",
            Self::June => "Jun",
            Self::July => "Jul",
            Self::August => "Aug",
            Self::September => "Sep",
            Self::October => "Oct",
            Self::November => "Nov",
            Self::December => "Dec",
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(try_from = "CalendarDateDocument")]
pub struct CalendarDate {
    year: u64,
    month: Month,
    day: u32,
}

impl CalendarDate {
    pub fn new(year: u64, month: Month, day: u32) -> Result<Self> {
        match year {
            0 => Err(anyhow!("calendar years start at 1")),
            _ => match (1..=month.days()).contains(&day) {
                true => Ok(Self { year, month, day }),
                false => Err(anyhow!("day is outside the selected month")),
            },
        }
    }

    pub fn on_day(elapsed_days: u64) -> Self {
        let day_of_year = (elapsed_days % u64::from(DAYS_PER_YEAR)) as u32;
        let month = match day_of_year {
            0..=30 => Month::January,
            31..=58 => Month::February,
            59..=89 => Month::March,
            90..=119 => Month::April,
            120..=150 => Month::May,
            151..=180 => Month::June,
            181..=211 => Month::July,
            212..=242 => Month::August,
            243..=272 => Month::September,
            273..=303 => Month::October,
            304..=333 => Month::November,
            _ => Month::December,
        };
        Self {
            year: elapsed_days / u64::from(DAYS_PER_YEAR) + 1,
            month,
            day: day_of_year - month.first_day() + 1,
        }
    }

    pub fn year(self) -> u64 {
        self.year
    }

    pub fn month(self) -> Month {
        self.month
    }

    pub fn day(self) -> u32 {
        self.day
    }

    pub fn day_of_year(self) -> u32 {
        self.month.first_day() + self.day - 1
    }
}

impl Default for CalendarDate {
    fn default() -> Self {
        Self::on_day(0)
    }
}

impl fmt::Display for CalendarDate {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:02} {} {:04}", self.day, self.month, self.year)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct CalendarDateDocument {
    year: u64,
    month: Month,
    day: u32,
}

impl TryFrom<CalendarDateDocument> for CalendarDate {
    type Error = anyhow::Error;

    fn try_from(document: CalendarDateDocument) -> Result<Self> {
        Self::new(document.year, document.month, document.day)
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Season {
    Spring,
    Summer,
    Autumn,
    Winter,
}

impl Season {
    pub fn on_date(date: CalendarDate) -> Self {
        match date.month() {
            Month::March | Month::April | Month::May => Self::Spring,
            Month::June | Month::July | Month::August => Self::Summer,
            Month::September | Month::October | Month::November => Self::Autumn,
            Month::December | Month::January | Month::February => Self::Winter,
        }
    }
}

impl fmt::Display for Season {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{self:?}")
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum HarvestSeason {
    Dormant,
    Growing,
    Harvesting,
}

impl HarvestSeason {
    pub fn on_date(date: CalendarDate) -> Self {
        Self::in_window(date, HARVEST_START, HARVEST_DAYS)
    }

    pub(super) fn in_window(date: CalendarDate, harvest_start: u32, harvest_days: u32) -> Self {
        let elapsed = (date.day_of_year() + DAYS_PER_YEAR - harvest_start) % DAYS_PER_YEAR;
        let growing_days = 211.min(DAYS_PER_YEAR - harvest_days);
        match elapsed {
            phase if phase < harvest_days => Self::Harvesting,
            phase if phase >= DAYS_PER_YEAR - growing_days => Self::Growing,
            _ => Self::Dormant,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Weather {
    pub date: CalendarDate,
    pub season: Season,
    pub temperature_celsius: i32,
    pub heating_units_per_person: u32,
    pub harvest: HarvestSeason,
}

impl Weather {
    pub fn on_day(elapsed_days: u64) -> Self {
        let date = CalendarDate::on_day(elapsed_days);
        let day_of_year = date.day_of_year();
        let temperature_celsius = match day_of_year {
            0..=182 => -5 + (day_of_year * 28 / 182) as i32,
            _ => 23 - ((day_of_year - 182) * 28 / 183) as i32,
        };
        Self {
            date,
            season: Season::on_date(date),
            temperature_celsius,
            heating_units_per_person: ((10 - temperature_celsius).max(0) as u32).div_ceil(5),
            harvest: HarvestSeason::on_date(date),
        }
    }
}

impl Default for Weather {
    fn default() -> Self {
        Self::on_day(0)
    }
}
