use std::{
    cmp::Ordering,
    collections::HashMap,
    fmt,
    hash::{Hash, Hasher},
    iter,
    marker::PhantomData,
    mem, ops, ptr,
    rc::Rc,
    sync::Arc,
};

use educe::Educe;
use either::{Either, Left, Right};
use itertools::Itertools;
use rustc_hash::FxHasher;
use serde::{
    de::{Deserializer, Error},
    Deserialize,
};
use typed_arena::Arena;

#[derive(Clone, Copy, Default, Debug, Educe)]
#[educe(
    PartialEq(bound = "W: PartialEq"),
    Eq(bound = "W: Eq"),
    PartialOrd(bound = "W: PartialOrd"),
    Ord(bound = "W: Ord"),
    Hash(bound = "W: Hash")
)]
pub struct WithOrdOf<W, T>(
    pub W,
    #[educe(PartialEq(ignore), PartialOrd(ignore), Ord(ignore), Hash(ignore))] pub T,
);

pub type AlwaysEq<T> = WithOrdOf<(), T>;

impl<T> AlwaysEq<T> {
    pub fn new(value: T) -> Self {
        WithOrdOf((), value)
    }
}

pub trait StableDeref: ops::Deref {}
impl<T: ?Sized> StableDeref for Box<T> {}
impl<T: ?Sized> StableDeref for Rc<T> {}
impl<T: ?Sized> StableDeref for Arc<T> {}
impl<T: ?Sized> StableDeref for &'_ T {}

#[derive(Clone, Copy, Debug)]
pub struct PtrOrd<T>(pub T);

impl<T: StableDeref> PtrOrd<T> {
    #[must_use]
    pub fn new<U>(value: U) -> Self
    where
        T: From<U>,
    {
        PtrOrd(value.into())
    }

    fn addr(&self) -> usize {
        ptr::addr_of!(*self.0).cast::<()>() as usize
    }
}

impl<T> PtrOrd<Box<T>> {
    #[must_use]
    pub fn into_inner(self) -> T {
        *self.0
    }
}

impl<T: StableDeref> PartialEq for PtrOrd<T> {
    fn eq(&self, other: &Self) -> bool {
        self.addr() == other.addr()
    }
}

impl<T: StableDeref> Eq for PtrOrd<T> {}

impl<T: StableDeref> PartialOrd for PtrOrd<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: StableDeref> Ord for PtrOrd<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.addr().cmp(&other.addr())
    }
}

impl<T: StableDeref> Hash for PtrOrd<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.addr().hash(state);
    }
}

impl<T: StableDeref> ops::Deref for PtrOrd<T> {
    type Target = T::Target;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: StableDeref + ops::DerefMut> ops::DerefMut for PtrOrd<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[must_use]
pub fn add_size_hints(
    l: (usize, Option<usize>),
    r: (usize, Option<usize>),
) -> (usize, Option<usize>) {
    (l.0 + r.0, l.1.zip(r.1).map(|(l, r)| l + r))
}

pub struct Lazy<T, F>(Either<Option<F>, T>);

impl<T, F: FnOnce() -> T> Lazy<T, F> {
    #[must_use]
    pub fn new(f: F) -> Self {
        Lazy(Left(Some(f)))
    }

    #[must_use]
    pub fn try_get(self) -> Option<T> {
        self.0.right()
    }

    #[must_use]
    pub fn try_ref(&self) -> Option<&T> {
        self.0.as_ref().right()
    }

    #[must_use]
    pub fn try_mut(&mut self) -> Option<&mut T> {
        self.0.as_mut().right()
    }

    #[must_use]
    pub fn get_mut(&mut self) -> &mut T {
        let out = mem::replace(self, Lazy(Left(None))).get();
        *self = Lazy(Right(out));
        self.0.as_mut().right().unwrap()
    }

    #[must_use]
    pub fn get(self) -> T {
        match self.0 {
            Left(init) => init.unwrap()(),
            Right(out) => out,
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Educe)]
#[educe(Default(expression = "SmallBitSetInner::Inline([0])"))]
enum SmallBitSetInner {
    Inline([usize; 1]),
    Alloc(Rc<[usize]>),
}

const USIZE_BITS: usize = usize::BITS as usize;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SmallBitSet(SmallBitSetInner);

impl SmallBitSet {
    fn with_len(len: usize) -> Self {
        SmallBitSet(if len <= 1 {
            SmallBitSetInner::Inline([0])
        } else {
            SmallBitSetInner::Alloc(iter::repeat(0).take(len).collect())
        })
    }

    #[must_use]
    pub fn with_capacity(cap: usize) -> Self {
        Self::with_len(cap.div_ceil(USIZE_BITS))
    }

    #[must_use]
    pub fn backing(&self) -> &[usize] {
        match self.0 {
            SmallBitSetInner::Inline(ref inner) => inner,
            SmallBitSetInner::Alloc(ref inner) => inner,
        }
    }

    pub fn backing_iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.backing().iter().copied()
    }

    #[must_use]
    pub fn backing_mut(&mut self) -> &mut [usize] {
        match self.0 {
            SmallBitSetInner::Inline(ref mut inner) => inner,
            SmallBitSetInner::Alloc(ref mut inner) => {
                if Rc::strong_count(inner) > 1 {
                    *inner = Rc::from(&**inner);
                }

                Rc::get_mut(inner).unwrap()
            },
        }
    }

    #[must_use]
    pub fn capacity(&self) -> usize {
        self.backing().len() * USIZE_BITS
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.backing_iter().all(|i| i == 0)
    }

    #[must_use]
    pub fn subset_of(&self, other: &Self) -> bool {
        use itertools::EitherOrBoth;

        self.backing_iter()
            .zip_longest(other.backing_iter())
            .all(|i| match i {
                EitherOrBoth::Both(l, r) => (l & !r) == 0,
                EitherOrBoth::Left(l) => l == 0,
                EitherOrBoth::Right(_) => true,
            })
    }

    #[must_use]
    pub fn intersects_with(&self, other: &Self) -> bool {
        self.backing_iter()
            .zip(other.backing_iter())
            .all(|(l, r)| l & r != 0)
    }

    #[allow(clippy::cast_possible_truncation)] // the truncation is correct (and unlikely, too)
    fn index(index: usize) -> (usize, usize) {
        (index / USIZE_BITS, 1usize << (index % USIZE_BITS))
    }

    #[must_use]
    pub fn bit(&self, at: usize) -> bool {
        let (idx, mask) = Self::index(at);
        self.backing()[idx] & mask != 0
    }

    pub fn set_bit_to(&mut self, value: bool, at: usize) {
        let (idx, mask) = Self::index(at);
        if value {
            self.backing_mut()[idx] |= mask;
        } else if let Some(x) = self.backing_mut().get_mut(idx) {
            *x &= !mask;
        }
    }

    pub fn set(&mut self, at: usize) {
        self.set_bit_to(true, at);
    }

    pub fn unset(&mut self, at: usize) {
        self.set_bit_to(true, at);
    }

    #[must_use]
    pub fn first_set(&self) -> Option<usize> {
        (0..self.capacity()).find(|&i| self.bit(i))
        /*
        let lower_bits = usize::BITS.ilog2();
        for (idx, i) in self.backing_iter().enumerate() {
            if i != 0 {
                return Some(idx << lower_bits | (i.trailing_zeros() as usize));
            }
        }
        None
        */
    }

    #[must_use]
    pub fn last_set(&self) -> Option<usize> {
        // TODO: optimize
        (0..self.capacity()).rev().find(|&i| self.bit(i))
    }

    fn resize_backing_len(&mut self, backing_len: usize) {
        if self.backing().len() < backing_len {
            // This can only happen in the case that we need the heap anyways
            self.0 = SmallBitSetInner::Alloc(
                self.backing_iter()
                    .chain(iter::repeat(0).take(backing_len - self.backing().len()))
                    .collect(),
            );
        }
    }

    pub fn resize(&mut self, capacity: usize) {
        self.resize_backing_len(capacity.div_ceil(USIZE_BITS));
    }

    pub fn resizing_set_to(&mut self, value: bool, at: usize) {
        if value {
            self.resizing_set(at);
        } else {
            self.unset(at);
        }
    }

    pub fn resizing_set(&mut self, at: usize) {
        self.resize(at + 1);
        self.set(at);
    }

    pub fn invert_existing(&mut self) {
        for i in self.backing_mut() {
            *i = !*i;
        }
    }

    pub fn invert(&mut self, capacity: usize) {
        self.resize(capacity);
        self.invert_existing();
    }

    // TODO: Check my maths & simplify potentially
    #[allow(clippy::cast_possible_truncation)] // we only care about the lower bits anyway
    fn shift_inner(
        &mut self,
        lower_by: usize,
        skip: usize,
        get: impl Fn(&[usize], usize, usize) -> Option<usize>,
    ) {
        let backing = self.backing_mut();
        let get = |backing: &_, idx| get(backing, idx, skip).unwrap_or(0);

        for i in 0..backing.len() {
            let mut out = get(backing, i).wrapping_shr(lower_by as u32);
            if lower_by % USIZE_BITS != 0 {
                out |= get(backing, i + 1).wrapping_shl(USIZE_BITS.wrapping_sub(lower_by) as u32);
            }
            backing[i] = out;
        }
    }

    fn shift_drop_inner(&mut self, by: usize, get: impl Fn(&[usize], usize) -> Option<usize>) {
        self.shift_inner(by, by / USIZE_BITS, |backing, idx, skip| {
            get(backing, idx + skip)
        });
    }

    // `ops::Shl` and `ops::Shr` could work but are slightly ambiguous through the
    // framing as not a bigint, but a bit set
    pub fn shift_drop_lower(&mut self, by: usize) {
        self.shift_drop_inner(by, |backing, idx| backing.get(idx).copied());
    }

    #[must_use]
    #[allow(clippy::cast_possible_truncation)] // we only care about the lower bits anyway
    pub fn shifted_dropped_lower(&self, by: usize) -> Self {
        let backing = self.backing();
        let dropped_last = by / USIZE_BITS
            + usize::from(
                by % USIZE_BITS != 0 && backing.last().unwrap().wrapping_shr(by as u32) == 0,
            );
        let mut out = Self::with_len(backing.len().saturating_sub(dropped_last));
        out.shift_drop_inner(by, |_, idx| backing.get(idx).copied());
        out
    }

    #[allow(clippy::cast_possible_truncation)] // we only care about the lower bits anyway
    fn shift_extend_len(&self, by: usize) -> usize {
        self.backing().len()
            + by / USIZE_BITS
            + usize::from(
                by % USIZE_BITS != 0
                    && self
                        .backing()
                        .last()
                        .unwrap()
                        .wrapping_shr(USIZE_BITS.wrapping_sub(by) as u32)
                        != 0,
            )
    }

    fn shift_extend_inner(
        &mut self,
        by: usize,
        fill: bool,
        get: impl Fn(&[usize], usize) -> Option<usize>,
    ) {
        self.shift_inner(
            (USIZE_BITS - (by % USIZE_BITS)) % USIZE_BITS,
            by / USIZE_BITS + usize::from(by % USIZE_BITS != 0),
            |backing, idx, skip| {
                idx.checked_sub(skip)
                    .map_or(Some(if fill { usize::MAX } else { 0 }), |idx| {
                        get(backing, idx)
                    })
            },
        );
    }

    pub fn shift_extend_lower(&mut self, by: usize, fill: bool) {
        self.resize_backing_len(self.shift_extend_len(by));
        self.shift_extend_inner(by, fill, |backing, idx| backing.get(idx).copied());
    }

    #[must_use]
    pub fn shifted_extended_lower(&self, by: usize, fill: bool) -> Self {
        let backing = self.backing();
        let mut out = Self::with_len(self.shift_extend_len(by));
        out.shift_extend_inner(by, fill, |_, idx| backing.get(idx).copied());
        out
    }
}

impl fmt::Debug for SmallBitSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            SmallBitSetInner::Inline(_) => write!(f, "[inline]")?,
            SmallBitSetInner::Alloc(_) => write!(f, "[alloc]")?,
        }

        for i in 0..self.capacity() {
            if self.bit(i) {
                write!(f, "1")?;
            } else {
                write!(f, "0")?;
            }
        }

        Ok(())
    }
}

impl Default for SmallBitSet {
    fn default() -> Self {
        SmallBitSet(SmallBitSetInner::Inline([0; 1]))
    }
}

impl ops::BitOr for SmallBitSet {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self {
        if rhs.backing().len() > self.backing().len() {
            rhs | self
        } else {
            self | &rhs
        }
    }
}

impl<'a> ops::BitOr<&'a SmallBitSet> for SmallBitSet {
    type Output = Self;

    fn bitor(mut self, rhs: &'a Self) -> Self {
        self.resize_backing_len(rhs.backing().len());

        for (out, rhs) in self.backing_mut().iter_mut().zip(rhs.backing_iter()) {
            *out |= rhs;
        }

        self
    }
}

impl ops::BitOr<SmallBitSet> for &'_ SmallBitSet {
    type Output = SmallBitSet;

    fn bitor(self, rhs: SmallBitSet) -> SmallBitSet {
        rhs | self
    }
}

impl<'a> ops::BitOr<&'a SmallBitSet> for &'_ SmallBitSet {
    type Output = SmallBitSet;

    fn bitor(self, rhs: &'a SmallBitSet) -> SmallBitSet {
        if self.backing().len() >= rhs.backing().len() {
            self.clone() | rhs
        } else {
            rhs.clone() | self
        }
    }
}

impl ops::BitOrAssign<SmallBitSet> for SmallBitSet {
    fn bitor_assign(&mut self, rhs: SmallBitSet) {
        *self = mem::take(self) | rhs;
    }
}

impl<'a> ops::BitOrAssign<&'a SmallBitSet> for SmallBitSet {
    fn bitor_assign(&mut self, rhs: &'a SmallBitSet) {
        *self = mem::take(self) | rhs;
    }
}

impl ops::BitAnd for SmallBitSet {
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self {
        if rhs.backing().len() > self.backing().len() {
            rhs & self
        } else {
            self & &rhs
        }
    }
}

impl<'a> ops::BitAnd<&'a SmallBitSet> for SmallBitSet {
    type Output = Self;

    fn bitand(mut self, rhs: &'a Self) -> Self {
        self.resize_backing_len(rhs.backing().len());

        for (out, rhs) in self.backing_mut().iter_mut().zip(rhs.backing_iter()) {
            *out &= rhs;
        }

        self
    }
}

impl ops::BitAnd<SmallBitSet> for &'_ SmallBitSet {
    type Output = SmallBitSet;

    fn bitand(self, rhs: SmallBitSet) -> SmallBitSet {
        rhs & self
    }
}

impl<'a> ops::BitAnd<&'a SmallBitSet> for &'_ SmallBitSet {
    type Output = SmallBitSet;

    fn bitand(self, rhs: &'a SmallBitSet) -> SmallBitSet {
        if self.backing().len() >= rhs.backing().len() {
            self.clone() & rhs
        } else {
            rhs.clone() & self
        }
    }
}

impl ops::BitAndAssign<SmallBitSet> for SmallBitSet {
    fn bitand_assign(&mut self, rhs: SmallBitSet) {
        *self = mem::take(self) & rhs;
    }
}

impl<'a> ops::BitAndAssign<&'a SmallBitSet> for SmallBitSet {
    fn bitand_assign(&mut self, rhs: &'a SmallBitSet) {
        *self = mem::take(self) & rhs;
    }
}

impl ops::BitXor for SmallBitSet {
    type Output = Self;

    fn bitxor(self, rhs: Self) -> Self {
        if rhs.backing().len() > self.backing().len() {
            rhs ^ self
        } else {
            self ^ &rhs
        }
    }
}

impl<'a> ops::BitXor<&'a SmallBitSet> for SmallBitSet {
    type Output = Self;

    fn bitxor(mut self, rhs: &'a Self) -> Self {
        self.resize_backing_len(rhs.backing().len());

        for (out, rhs) in self.backing_mut().iter_mut().zip(rhs.backing_iter()) {
            *out ^= rhs;
        }

        self
    }
}

impl ops::BitXor<SmallBitSet> for &'_ SmallBitSet {
    type Output = SmallBitSet;

    fn bitxor(self, rhs: SmallBitSet) -> SmallBitSet {
        rhs ^ self
    }
}

impl<'a> ops::BitXor<&'a SmallBitSet> for &'_ SmallBitSet {
    type Output = SmallBitSet;

    fn bitxor(self, rhs: &'a SmallBitSet) -> SmallBitSet {
        if self.backing().len() >= rhs.backing().len() {
            self.clone() ^ rhs
        } else {
            rhs.clone() ^ self
        }
    }
}

impl ops::BitXorAssign<SmallBitSet> for SmallBitSet {
    fn bitxor_assign(&mut self, rhs: SmallBitSet) {
        *self = mem::take(self) ^ rhs;
    }
}

impl<'a> ops::BitXorAssign<&'a SmallBitSet> for SmallBitSet {
    fn bitxor_assign(&mut self, rhs: &'a SmallBitSet) {
        *self = mem::take(self) ^ rhs;
    }
}

impl ops::Not for SmallBitSet {
    type Output = Self;

    fn not(mut self) -> Self {
        self.invert_existing();
        self
    }
}

impl ops::Not for &'_ SmallBitSet {
    type Output = SmallBitSet;

    fn not(self) -> SmallBitSet {
        !self.clone()
    }
}

#[derive(Educe)]
#[educe(
    Clone(bound = "O: Clone"),
    PartialEq(bound = "O: PartialEq"),
    Eq(bound = "O: Eq"),
    PartialOrd(bound = "O: PartialOrd"),
    Ord(bound = "O: Ord"),
    Hash(bound = "O: Hash"),
    Default(bound = "O: Default"),
    Debug(bound = "O: fmt::Debug")
)]
pub struct DeserializeVia<T, O>(pub O, PhantomData<fn(T)>);

impl<T, O: Copy> Copy for DeserializeVia<T, O> {}

impl<'de, O: TryFrom<T>, T: Deserialize<'de>> Deserialize<'de> for DeserializeVia<T, O>
where
    O::Error: fmt::Display,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(DeserializeVia(
            O::try_from(T::deserialize(deserializer)?).map_err(D::Error::custom)?,
            PhantomData,
        ))
    }
}

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum UntaggedEither<L, R> {
    Left(L),
    Right(R),
}

pub trait RefListExtra<T>: Copy {
    #[must_use]
    fn first(value: &T) -> Self;
    #[must_use]
    fn add(&self, value: &T) -> Self;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BasicRefList;

impl<T> RefListExtra<T> for BasicRefList {
    fn first(_: &T) -> Self {
        BasicRefList
    }

    fn add(&self, _: &T) -> Self {
        BasicRefList
    }
}

#[derive(Educe, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
#[educe(Clone(bound = "'static: 'a"), Copy(bound = "'static: 'a"))]
pub enum RefList<'a, T, H: RefListExtra<T> = BasicRefList> {
    Beginning,
    AddAtEnd(H, &'a (RefList<'a, T, H>, T)),
}

impl<T> RefList<'_, T> {
    #[must_use]
    pub fn new_basic() -> Self {
        RefList::Beginning
    }
}

impl<'a, T, H: RefListExtra<T>> RefList<'a, T, H> {
    pub fn rev_iter(mut self) -> impl Iterator<Item = &'a T> {
        iter::from_fn(move || match self {
            RefList::Beginning => None,
            RefList::AddAtEnd(_, &(before, ref last)) => {
                self = before;
                Some(last)
            },
        })
    }

    pub fn add(&mut self, value: T, arena: &'a Arena<(Self, T)>) {
        let extra = match *self {
            RefList::Beginning => H::first(&value),
            RefList::AddAtEnd(ref prev, _) => prev.add(&value),
        };
        *self = RefList::AddAtEnd(extra, arena.alloc((*self, value)));
    }

    #[must_use]
    pub fn collect(self) -> Vec<T>
    where
        T: Clone,
    {
        let mut out = self.rev_iter().cloned().collect::<Vec<_>>();
        out.reverse();
        out
    }
}

fn quick_hash<T: Hash>(x: &T) -> u64 {
    let mut hasher = FxHasher::default();
    x.hash(&mut hasher);
    hasher.finish()
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct HashRefList(u64);

impl<T: Hash> RefListExtra<T> for HashRefList {
    fn first(value: &T) -> Self {
        HashRefList(quick_hash(value))
    }

    fn add(&self, value: &T) -> Self {
        HashRefList(quick_hash(&(self.0, value)))
    }
}

impl<T: Hash> RefList<'_, T, HashRefList> {
    #[must_use]
    pub fn new_quickhash() -> Self {
        RefList::Beginning
    }
}

impl<T: Hash> Hash for RefList<'_, T, HashRefList> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            RefList::Beginning => state.write_u8(0),
            RefList::AddAtEnd(HashRefList(h), _) => {
                state.write_u8(1);
                state.write_u64(h);
            },
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum DeFail {}

#[allow(unused_tuple_struct_fields, clippy::zero_sized_map_values)]
#[derive(Deserialize)]
#[serde(untagged)]
enum TomlOptionDe<T> {
    Some(T),
    None(HashMap<String, DeFail>),
    Some2(T), // to make sure we get that error if possible
}

#[derive(Clone, Copy, Deserialize, Debug, Educe)]
#[serde(from = "TomlOptionDe<T>")]
#[educe(Default)]
pub struct TomlOption<T>(pub Option<T>);

impl<T> From<TomlOptionDe<T>> for TomlOption<T> {
    fn from(value: TomlOptionDe<T>) -> Self {
        match value {
            TomlOptionDe::Some(out) | TomlOptionDe::Some2(out) => TomlOption(Some(out)),
            TomlOptionDe::None(_) => TomlOption(None),
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct NoCopy<T>(pub T);

impl<T> NoCopy<T> {
    pub fn consume(self) -> T {
        self.0
    }
}

#[macro_export]
macro_rules! let_group {
    {let $group:ident: $groupty:ident; $(let $var:ident = $x:expr);* $(;)?} => {
        #[allow(non_camel_case_types)]
        struct $groupty<$($var = ()),*> {
            $($var: $var,)*
        }

        let $group = {
            $(let $var = $x;)*
            Group {
                $($var,)*
            }
        };
    };
    {let $group:ident; $(let $var:ident = $x:expr);* $(;)?} => {
        $crate::helpers::let_group!{let $group: Group; $(let $var = $x;)*}
    };
}

pub use crate::let_group;

#[must_use]
#[allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
pub fn parse_size(mut s: &str) -> Option<usize> {
    s = s.trim();

    if s == "none" {
        return None;
    }

    let suffixes = [("K", 1024), ("M", 1024 * 1024), ("G", 1024 * 1024 * 1024)];

    let mut multiplier = 1u32;
    for (suffix, possible_multiplier) in suffixes {
        if let Some(prefix) = s.strip_suffix(suffix) {
            s = prefix;
            multiplier = possible_multiplier;
            break;
        }
    }

    Some((s.parse::<f32>().unwrap() * multiplier as f32) as usize)
}
