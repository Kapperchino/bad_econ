use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub struct Position {
    pub x: u32,
    pub y: u32,
}
impl Position {
    pub fn distance(self, other: Self) -> u32 {
        self.x.abs_diff(other.x) + self.y.abs_diff(other.y)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Actor {
    pub identity: u64,
    pub position: Position,
    pub kind: u32,
    pub owner: u32,
    pub capacity: u32,
}
#[derive(Clone, Debug, Default, Serialize, PartialEq, Eq)]
pub struct ActorRows {
    identity: Vec<u64>,
    x: Vec<u32>,
    y: Vec<u32>,
    kind: Vec<u32>,
    owner: Vec<u32>,
    capacity: Vec<u32>,
    #[serde(skip)]
    firms: usize,
    #[serde(skip)]
    first_firm: usize,
}
impl ActorRows {
    #[inline]
    pub fn len(&self) -> usize {
        self.identity.len()
    }
    pub fn is_empty(&self) -> bool {
        self.identity.is_empty()
    }
    pub(crate) fn identities(&self) -> &[u64] {
        &self.identity
    }
    pub(crate) fn assigned_identity(&self, assignment: u64) -> Option<u64> {
        let index = usize::try_from(assignment.checked_sub(1)?).ok()?;
        self.identity.get(index).copied()
    }
    #[cfg(feature = "compute")]
    pub(crate) fn clone_for_growth(&self, additions: usize) -> Self {
        fn column<T: Copy>(values: &[T], capacity: usize) -> Vec<T> {
            let mut column = Vec::with_capacity(capacity);
            column.extend_from_slice(values);
            column
        }
        let capacity = self.len() + additions;
        Self {
            identity: column(&self.identity, capacity),
            x: column(&self.x, capacity),
            y: column(&self.y, capacity),
            kind: column(&self.kind, capacity),
            owner: column(&self.owner, capacity),
            capacity: column(&self.capacity, capacity),
            firms: self.firms,
            first_firm: self.first_firm,
        }
    }
    #[cfg(all(test, feature = "compute"))]
    pub(crate) fn growth_capacity(&self) -> usize {
        [
            self.identity.capacity(),
            self.x.capacity(),
            self.y.capacity(),
            self.kind.capacity(),
            self.owner.capacity(),
            self.capacity.capacity(),
        ]
        .into_iter()
        .min()
        .unwrap_or(0)
    }
    #[cfg(test)]
    pub(crate) fn rental_prefix_matches(&self, current: &Self) -> bool {
        [
            current.x.get(..self.len()) == Some(self.x.as_slice()),
            current.y.get(..self.len()) == Some(self.y.as_slice()),
            current.kind.get(..self.len()) == Some(self.kind.as_slice()),
            current.owner.get(..self.len()) == Some(self.owner.as_slice()),
        ]
        .into_iter()
        .all(std::convert::identity)
    }
    #[cfg(test)]
    pub(crate) fn rental_changes(&self, current: &Self) -> Vec<usize> {
        match self.rental_prefix_matches(current) {
            true => Vec::new(),
            false => (0..self.len().min(current.len()))
                .filter(|&index| {
                    self.x[index] != current.x[index]
                        || self.y[index] != current.y[index]
                        || self.kind[index] != current.kind[index]
                        || self.owner[index] != current.owner[index]
                })
                .collect(),
        }
    }
    #[cfg(feature = "compute")]
    pub(crate) fn firms(&self) -> usize {
        self.firms
    }
    #[inline]
    pub fn at(&self, index: usize) -> Actor {
        Actor {
            identity: self.identity[index],
            position: Position {
                x: self.x[index],
                y: self.y[index],
            },
            kind: self.kind[index],
            owner: self.owner[index],
            capacity: self.capacity[index],
        }
    }
    pub fn get(&self, index: usize) -> Option<Actor> {
        (index < self.len()).then(|| self.at(index))
    }
    pub fn set(&mut self, index: usize, actor: Actor) {
        self.firms -= usize::from(self.kind[index] > 0);
        self.firms += usize::from(actor.kind > 0);
        self.identity[index] = actor.identity;
        self.x[index] = actor.position.x;
        self.y[index] = actor.position.y;
        self.kind[index] = actor.kind;
        self.owner[index] = actor.owner;
        self.capacity[index] = actor.capacity;
        self.first_firm = match (actor.kind, index.cmp(&self.first_firm)) {
            (0, std::cmp::Ordering::Equal) => self.kind[index + 1..]
                .iter()
                .position(|&kind| kind > 0)
                .map_or(self.len(), |offset| index + 1 + offset),
            (0, _) => self.first_firm,
            _ => self.first_firm.min(index),
        };
    }
    pub(crate) fn extend(&mut self, actors: impl IntoIterator<Item = Actor>) {
        let start = self.len();
        let actors = actors.into_iter();
        let additional = actors.size_hint().0;
        self.identity.reserve(additional);
        self.x.reserve(additional);
        self.y.reserve(additional);
        self.kind.reserve(additional);
        self.owner.reserve(additional);
        self.capacity.reserve(additional);
        for actor in actors {
            self.firms += usize::from(actor.kind > 0);
            self.identity.push(actor.identity);
            self.x.push(actor.position.x);
            self.y.push(actor.position.y);
            self.kind.push(actor.kind);
            self.owner.push(actor.owner);
            self.capacity.push(actor.capacity);
        }
        self.first_firm = match self.first_firm < start {
            true => self.first_firm,
            false => self.kind[start..]
                .iter()
                .position(|&kind| kind > 0)
                .map_or(self.len(), |offset| start + offset),
        };
    }
    #[inline]
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Actor> + '_ {
        self.iter_from(0)
    }
    #[inline]
    pub(crate) fn iter_from(&self, start: usize) -> impl ExactSizeIterator<Item = Actor> + '_ {
        (start..self.len()).map(|index| self.at(index))
    }
    #[cfg(feature = "compute")]
    pub(crate) fn transfer_firm_ownership(&mut self, previous: u32, successor: u32) {
        for (owner, &kind) in self.owner[self.first_firm..]
            .iter_mut()
            .zip(&self.kind[self.first_firm..])
        {
            match (kind, *owner) {
                (1.., current) if current == previous => *owner = successor,
                _ => (),
            }
        }
    }

    pub(crate) fn firm_indices(&self) -> impl Iterator<Item = usize> + '_ {
        let start = self.first_firm;
        self.kind[start..]
            .iter()
            .enumerate()
            .filter_map(move |(offset, &kind)| (kind > 0).then_some(start + offset))
    }
    pub(crate) fn firm_actors(&self) -> impl Iterator<Item = Actor> + '_ {
        self.firm_indices().map(|index| self.at(index))
    }
}
impl FromIterator<Actor> for ActorRows {
    fn from_iter<T: IntoIterator<Item = Actor>>(actors: T) -> Self {
        let mut rows = Self::default();
        rows.extend(actors);
        rows
    }
}
impl<'de> Deserialize<'de> for ActorRows {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Rows {
            identity: Vec<u64>,
            x: Vec<u32>,
            y: Vec<u32>,
            kind: Vec<u32>,
            owner: Vec<u32>,
            capacity: Vec<u32>,
        }
        let rows = Rows::deserialize(deserializer)?;
        match [
            rows.x.len(),
            rows.y.len(),
            rows.kind.len(),
            rows.owner.len(),
            rows.capacity.len(),
        ]
        .into_iter()
        .all(|len| len == rows.identity.len())
        {
            true => Ok(Self {
                firms: rows.kind.iter().filter(|&&kind| kind > 0).count(),
                first_firm: rows
                    .kind
                    .iter()
                    .position(|&kind| kind > 0)
                    .unwrap_or(rows.kind.len()),
                identity: rows.identity,
                x: rows.x,
                y: rows.y,
                kind: rows.kind,
                owner: rows.owner,
                capacity: rows.capacity,
            }),
            false => Err(serde::de::Error::custom(
                "actor rows have different lengths",
            )),
        }
    }
}
