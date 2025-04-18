use std::{collections::HashSet, hash::Hash, iter, mem, ops, rc::Rc, str::FromStr, sync::Arc};

use educe::Educe;
use either::{Either, Left, Right};
use num_traits::AsPrimitive;
use serde::Deserialize;

pub use crate::ipa::Element as IpaElement;
use crate::{
    explore::{self, Explorer},
    helpers::{add_size_hints, Lazy, NoCopy, PtrOrd},
};

pub type OrthoElement = char;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Instruction<Item> {
    Item { narrow: Item, consume: bool },
    Skip { branch: bool, count: usize },
}

impl<Item> Instruction<Item> {
    const NO_OP: Self = Instruction::Skip {
        branch: false,
        count: 0,
    };
}

impl<Item> Default for Instruction<Item> {
    fn default() -> Self {
        Self::NO_OP
    }
}

#[derive(Debug)]
enum Ast<Item> {
    Item { narrow: Item, consume: bool },
    Alt(Vec<Ast<Item>>),
    Opt(Box<Ast<Item>>),
    Seq(Vec<Ast<Item>>),
    Ghost(Box<Ast<Item>>),
}

impl<Item> Ast<Item> {
    fn map_item_inner<O>(self, f: &impl Fn(Item) -> O) -> Ast<O> {
        let map_vec = |v: Vec<Self>| v.into_iter().map(|i| i.map_item_inner(f)).collect();

        match self {
            Ast::Item { narrow, consume } => Ast::Item {
                narrow: f(narrow),
                consume,
            },
            Ast::Alt(alts) => Ast::Alt(map_vec(alts)),
            Ast::Opt(opt) => Ast::Opt(Box::new(opt.map_item_inner(f))),
            Ast::Seq(alts) => Ast::Seq(map_vec(alts)),
            Ast::Ghost(opt) => Ast::Ghost(Box::new(opt.map_item_inner(f))),
        }
    }

    fn map_item<O>(self, f: impl Fn(Item) -> O) -> Ast<O> {
        self.map_item_inner(&f)
    }

    fn alts(self, out: &mut Vec<Self>, optional: &mut bool, consume: bool) {
        match self {
            Ast::Alt(alts) => {
                out.reserve(alts.len());
                for i in alts {
                    i.alts(out, optional, consume);
                }
            },
            Ast::Opt(inner) => {
                *optional = true;
                inner.alts(out, optional, consume);
            },
            this => match this.simplify_inner(consume) {
                again @ (Ast::Alt(_) | Ast::Opt(_)) => again.alts(out, optional, consume),
                Ast::Seq(v) if v.is_empty() => *optional = true,
                simplified => out.push(simplified),
            },
        }
    }

    fn seqs(self, out: &mut Vec<Self>, consume: bool) -> bool {
        match self {
            Ast::Seq(seq) => {
                out.reserve(seq.len());
                for i in seq {
                    if !i.seqs(out, consume) {
                        return false;
                    }
                }

                true
            },
            this => match this.simplify_inner(consume) {
                again @ Ast::Seq(_) => again.seqs(out, consume),
                Ast::Alt(alts) if alts.is_empty() => false,
                simplified => {
                    out.push(simplified);
                    true
                },
            },
        }
    }

    fn simplify_inner(self, consume: bool) -> Self {
        match self {
            Ast::Item {
                narrow,
                consume: inner_consume,
            } => Ast::Item {
                narrow,
                consume: consume && inner_consume,
            },
            alts @ (Ast::Alt(_) | Ast::Opt(_)) => {
                let mut out = Vec::new();
                let mut optional = false;
                alts.alts(&mut out, &mut optional, consume);
                match (out.len(), optional) {
                    (0, false) => Ast::Alt(Vec::new()),
                    (0, true) => Ast::Seq(Vec::new()),
                    (1, false) => out.pop().unwrap(),
                    (1, true) => Ast::Opt(Box::new(out.pop().unwrap())),
                    (_, false) => Ast::Alt(out),
                    (_, true) => Ast::Opt(Box::new(Ast::Alt(out))),
                }
            },
            seq @ Ast::Seq(_) => {
                let mut out = Vec::new();
                if seq.seqs(&mut out, consume) {
                    if out.len() == 1 {
                        out.pop().unwrap()
                    } else {
                        Ast::Seq(out)
                    }
                } else {
                    Ast::Alt(Vec::new())
                }
            },
            Ast::Ghost(ghost) => ghost.simplify_inner(false),
        }
    }

    fn simplify(self) -> Self {
        self.simplify_inner(true)
    }

    fn write_instructions_inner(self, out: &mut Vec<Instruction<Item>>, consume: bool) {
        let backref = |v: &mut Vec<_>| {
            v.push(Instruction::default());
            v.len() - 1
        };

        let skip_to_here_from = |idx: usize, v: &mut Vec<_>, branch| {
            v[idx] = Instruction::Skip {
                branch,
                count: v.len() - idx - 1,
            };
        };

        match self {
            Ast::Item {
                narrow,
                consume: consume_inner,
            } => out.push(Instruction::Item {
                narrow,
                consume: consume_inner && consume,
            }),
            Ast::Alt(alts) if alts.is_empty() => out.push(Instruction::default()),
            Ast::Alt(alts) => {
                let mut last_branch = None;
                let mut skips = Vec::new();

                let last_idx = alts.len() - 1;
                for (idx, i) in alts.into_iter().enumerate() {
                    if let Some(last_branch) = last_branch {
                        skip_to_here_from(last_branch, out, true);
                    }
                    last_branch = (idx != last_idx).then(|| backref(out));

                    i.write_instructions_inner(out, consume);

                    if idx != last_idx {
                        skips.push(backref(out));
                    }
                }

                for i in skips {
                    skip_to_here_from(i, out, false);
                }
            },
            Ast::Opt(inner) => {
                let idx = backref(out);
                inner.write_instructions_inner(out, consume);
                skip_to_here_from(idx, out, true);
            },
            Ast::Seq(seqs) => {
                for i in seqs {
                    i.write_instructions_inner(out, consume);
                }
            },
            Ast::Ghost(ghost) => ghost.write_instructions_inner(out, false),
        }
    }

    fn into_instructions(self) -> Vec<Instruction<Item>> {
        let mut out = Vec::new();
        self.write_instructions_inner(&mut out, true);
        out
    }
}

mod parser {
    use nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::{char, none_of},
        combinator::{all_consuming, consumed, not, verify},
        error::Error as E,
        multi::{many0, many0_count, separated_list1},
        sequence::{delimited, preceded},
        Finish, IResult, Parser,
    };

    #[allow(clippy::wildcard_imports)]
    use super::*;
    use crate::ipa::parser as ipa_parser;

    const NOT_ALLOWED: &str = ".(?|)#[]";

    pub(super) fn ortho_element(s: &str) -> IResult<&str, char> {
        none_of(NOT_ALLOWED).parse(s)
    }

    pub(super) fn ipa_element(s: &str) -> IResult<&str, IpaElement> {
        verify(consumed(ipa_parser::parse_element), |&(s2, _)| {
            s2.chars().all(|i| !NOT_ALLOWED.contains(i))
        })
        .map(|(_, out)| out)
        .parse(s)
    }

    pub(super) fn context<'a, I>(
        element: impl Fn(&'a str) -> IResult<&'a str, I>,
    ) -> impl Fn(&'a str) -> IResult<&'a str, Ast<Option<I>>> {
        let make_item = |narrow| Ast::Item {
            narrow,
            consume: true,
        };
        let element = move |s| (&element).map(|x| make_item(Some(x))).parse(s);

        move |s| {
            alt((
                char('.').map(|_| make_item(None)),
                delimited(char('['), many0(&element).map(Ast::Alt), char(']')),
                &element,
            ))
            .parse(s)
        }
    }

    pub(super) fn alternate_context<'a, A, B>(
        primary_element: impl Fn(&'a str) -> IResult<&'a str, A>,
        secondary_element: impl Fn(&'a str) -> IResult<&'a str, B>,
    ) -> impl Fn(&'a str) -> IResult<&'a str, Ast<Either<Option<A>, Option<B>>>> {
        let primary_element = context(primary_element);
        let secondary_element = context(secondary_element);

        move |s| {
            alt((
                preceded(
                    char('#'),
                    alt((
                        (&secondary_element).map(|x| x.map_item(Right)),
                        delimited(
                            char('('),
                            parser_for(alt_item, &secondary_element).map(|x| x.map_item(Right)),
                            char(')'),
                        ),
                    )),
                ),
                (&primary_element).map(|x| x.map_item(Left)),
            ))
            .parse(s)
        }
    }

    fn parser_for<'a, 'b, C, I>(
        f: impl Fn(&C, &'a str) -> IResult<&'a str, Ast<I>> + 'b,
        c: &'b C,
    ) -> impl Parser<&'a str, Ast<I>, E<&'a str>> + 'b {
        move |s| f(c, s)
    }

    fn atom<'a, I>(
        item: &impl Fn(&'a str) -> IResult<&'a str, Ast<I>>,
        s: &'a str,
    ) -> IResult<&'a str, Ast<I>> {
        alt((
            delimited(tag("(?:"), parser_for(alt_item, item), char(')'))
                .map(|inner| Ast::Ghost(Box::new(inner))),
            delimited(
                char('('),
                preceded(not(char('?')), parser_for(alt_item, item)),
                char(')'),
            ),
            item,
        ))
        .parse(s)
    }

    fn opt_item<'a, I>(
        c: &impl Fn(&'a str) -> IResult<&'a str, Ast<I>>,
        s: &'a str,
    ) -> IResult<&'a str, Ast<I>> {
        parser_for(atom, c)
            .and(many0_count(char('?')))
            .map(|(mut out, optional_count)| {
                for _ in 0..optional_count {
                    out = Ast::Opt(Box::new(out));
                }
                out
            })
            .parse(s)
    }

    fn seq_item<'a, I>(
        c: &impl Fn(&'a str) -> IResult<&'a str, Ast<I>>,
        s: &'a str,
    ) -> IResult<&'a str, Ast<I>> {
        many0(parser_for(opt_item, c)).map(Ast::Seq).parse(s)
    }

    fn alt_item<'a, I>(
        c: &impl Fn(&'a str) -> IResult<&'a str, Ast<I>>,
        s: &'a str,
    ) -> IResult<&'a str, Ast<I>> {
        separated_list1(char('|'), parser_for(seq_item, c))
            .map(Ast::Alt)
            .parse(s)
    }

    pub(super) fn parse<'a, I>(
        ctx: impl Fn(&'a str) -> IResult<&'a str, Ast<I>>,
        s: &'a str,
    ) -> Result<Matcher<I>, String> {
        let parsed = all_consuming(parser_for(alt_item, &ctx))
            .parse(s)
            .finish()
            .map(|(_, out)| out);

        parsed
            .map(|out| Matcher(out.simplify().into_instructions()))
            .map_err(|err| {
                let idx = s.len() - err.input.len();

                let mut out = String::with_capacity(s.len() + 1 + idx + 1);

                out.push_str(s);

                out.push('\n');
                for _ in 0..idx {
                    out.push(' ');
                }
                out.push('^');

                out
            })
    }
}

#[derive(Educe)]
#[educe(Clone(bound = "'static: 'a"), Copy(bound = "'static: 'a"))]
// weird bounds because of the single_use_lifetimes warning
struct StatItem<'a, Item> {
    consume: bool,
    narrow: &'a Item,
}

impl<'a, Item> StatItem<'a, Item> {
    fn map_narrow<'b, O>(self, f: impl Fn(&'a Item) -> &'b O) -> StatItem<'b, O> {
        StatItem {
            consume: self.consume,
            narrow: f(self.narrow),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub enum InputBranching {
    #[default]
    None,
    NoDeduplication {
        const_depth: bool,
    },
    /// This deduplicates for `(depth, before, item, after)`; so this is only
    /// useful if those four often coincide
    WithDeduplication {
        const_depth: bool,
    },
}

pub trait Input {
    type State: Clone + Eq + Hash;
    type Item: Clone + Eq + Hash;

    const BRANCHING: InputBranching;

    /// Returns the initial state for which `next_with` will return the first
    /// item(s) (unless the input is empty)
    #[must_use]
    fn initial(&self) -> Self::State;

    /// Returns a final state for which `next_with` will return no items
    #[must_use]
    fn possible_final(&self) -> Self::State;

    /// Return all the possible continuations of the given state together with
    /// their first items
    #[must_use]
    fn possible_nexts_with<'a>(
        &self,
        state: Self::State,
    ) -> impl Iterator<Item = (Self::State, Self::Item)> + Clone + 'a
    where
        Self: 'a;

    #[must_use]
    fn monotonic_size_hint(&self, _state: &Self::State) -> (usize, Option<usize>) {
        (0, None)
    }

    #[must_use]
    fn branching_size_hint(&self, state: &Self::State) -> (usize, Option<usize>) {
        if Self::BRANCHING == InputBranching::None {
            (self.monotonic_size_hint(state).0, None)
        } else {
            self.monotonic_size_hint(state)
        }
    }

    #[must_use]
    fn monotonic_iter(&self, state: Self::State) -> InputMonotonicIter<Self> {
        InputMonotonicIter(self, state)
    }

    #[must_use]
    fn branching_iter(&self, state: Self::State) -> InputBranchingIter<Self> {
        InputBranchingIter {
            input: self,
            current_state: (0, state),
            todo: Vec::new(),
            seen: HashSet::new(),
        }
    }
}

macro_rules! impl_deref_matcher_input {
    ($(($($generics:tt)*))? for $t:ty, via $u:ty) => {
        impl $(<$($generics)*>)? Input for $t {
            type Item = <$u as Input>::Item;
            type State = <$u as Input>::State;

            const BRANCHING: InputBranching = <$u as Input>::BRANCHING;

            fn initial(&self) -> Self::State {
                <$u as Input>::initial(self)
            }

            fn possible_final(&self) -> Self::State {
                <$u as Input>::possible_final(self)
            }

            fn possible_nexts_with<'a>(
                &self,
                state: Self::State,
            ) -> impl Iterator<Item = (Self::State, Self::Item)> + Clone + 'a
            where Self: 'a {
                <$u as Input>::possible_nexts_with(self, state)
            }

            fn monotonic_size_hint(&self, state: &Self::State) -> (usize, Option<usize>) {
                <$u as Input>::monotonic_size_hint(self, state)
            }

            fn branching_size_hint(&self, state: &Self::State) -> (usize, Option<usize>) {
                <$u as Input>::branching_size_hint(self, state)
            }
        }
    }
}

impl_deref_matcher_input!((T: Input + ?Sized) for &'_ T, via T);
impl_deref_matcher_input!((T: Input + ?Sized) for Box<T>, via T);
impl_deref_matcher_input!((T: Input + ?Sized) for Rc<T>, via T);
impl_deref_matcher_input!((T: Input + ?Sized) for Arc<T>, via T);

impl_deref_matcher_input!((T: Input + ?Sized) for PtrOrd<&'_ T>, via T);
impl_deref_matcher_input!((T: Input + ?Sized) for PtrOrd<Box<T>>, via T);
impl_deref_matcher_input!((T: Input + ?Sized) for PtrOrd<Rc<T>>, via T);
impl_deref_matcher_input!((T: Input + ?Sized) for PtrOrd<Arc<T>>, via T);

impl Input for str {
    type Item = char;
    type State = usize;

    const BRANCHING: InputBranching = InputBranching::None;

    fn initial(&self) -> usize {
        0
    }

    fn possible_final(&self) -> Self::State {
        self.len()
    }

    fn possible_nexts_with<'a>(
        &self,
        state: usize,
    ) -> impl Iterator<Item = (usize, char)> + Clone + 'a {
        self[state.min(self.len())..]
            .chars()
            .next()
            .map(|out| (state + out.len_utf8(), out))
            .into_iter()
    }

    fn monotonic_size_hint(&self, state: &usize) -> (usize, Option<usize>) {
        let len = self.len() - *state;
        (len.div_ceil(4), Some(len))
    }
}

impl_deref_matcher_input!(for String, via str);

impl<T: Clone + Eq + Hash> Input for [T] {
    type Item = T;
    type State = usize;

    const BRANCHING: InputBranching = InputBranching::None;

    fn initial(&self) -> usize {
        0
    }

    fn possible_final(&self) -> Self::State {
        self.len()
    }

    fn possible_nexts_with<'a>(&self, state: usize) -> impl Iterator<Item = (usize, T)> + Clone + 'a
    where
        T: 'a,
    {
        self.get(state)
            .map(|out| (state + 1, out.clone()))
            .into_iter()
    }

    fn monotonic_size_hint(&self, state: &usize) -> (usize, Option<usize>) {
        let len = self.len() - *state;
        (len, Some(len))
    }
}

impl_deref_matcher_input!((T: Clone + Eq + Hash) for Vec<T>, via [T]);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct InputIterItem<State, Item> {
    pub before: State,
    pub item: Item,
    pub after: State,
}

#[derive(Educe)]
#[educe(Clone(bound = "'static: 'a"))] // weird bound because of single use lifetime warning
pub struct InputMonotonicIter<'a, I: Input + ?Sized>(&'a I, I::State);

impl<'a, I: Input + ?Sized> InputMonotonicIter<'a, I> {
    #[must_use]
    pub fn input(&self) -> &'a I {
        self.0
    }

    #[must_use]
    pub fn state(self) -> I::State {
        self.1
    }

    #[must_use]
    pub fn state_ref(&self) -> &I::State {
        &self.1
    }

    #[must_use]
    pub fn branching_from_here(self) -> InputBranchingIter<'a, I> {
        self.0.branching_iter(self.1)
    }
}

impl<I: Input + ?Sized> Iterator for InputMonotonicIter<'_, I> {
    type Item = InputIterItem<I::State, I::Item>;

    fn next(&mut self) -> Option<Self::Item> {
        self.0
            .possible_nexts_with(self.1.clone())
            .next()
            .map(|(after, item)| {
                let before = mem::replace(&mut self.1, after);
                InputIterItem {
                    before,
                    item,
                    after: self.1.clone(),
                }
            })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.monotonic_size_hint(&self.1)
    }

    fn count(mut self) -> usize {
        let hint = self.size_hint();

        if Some(hint.0) == hint.1 {
            hint.0
        } else {
            let mut out = 0;
            while self.next().is_some() {
                out += 1;
            }

            out
        }
    }
}

type BranchingIterItem<State, Item> = (usize, InputIterItem<State, Item>);

#[derive(Educe)]
#[educe(Clone(bound = "'static: 'a"))] // weird bound because of single use lifetime warning
pub struct InputBranchingIter<'a, I: Input + ?Sized> {
    input: &'a I,
    current_state: (usize, I::State),
    todo: Vec<BranchingIterItem<I::State, I::Item>>,
    seen: HashSet<BranchingIterItem<I::State, I::Item>>,
}

impl<I: Input + ?Sized> Iterator for InputBranchingIter<'_, I> {
    type Item = (usize, InputIterItem<I::State, I::Item>);

    fn next(&mut self) -> Option<Self::Item> {
        let mut nexts = self
            .input
            .possible_nexts_with(self.current_state.1.clone())
            .map(|(after, item)| {
                (self.current_state.0, InputIterItem {
                    before: self.current_state.1.clone(),
                    item,
                    after,
                })
            })
            .filter(|i| {
                if !matches!(I::BRANCHING, InputBranching::WithDeduplication { .. }) {
                    true
                } else {
                    self.seen.insert(i.clone())
                }
            });

        let out = match nexts.next() {
            None => match self.todo.pop() {
                None => return None,
                Some(out) => out,
            },
            Some(out) => {
                self.todo.extend(nexts);
                out
            },
        };

        self.current_state = (out.0 + 1, out.1.after.clone());

        Some(out)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min, max) = self
            .todo
            .iter()
            .map(|i| self.input.branching_size_hint(&i.1.after))
            .fold(
                self.input.branching_size_hint(&self.current_state.1),
                add_size_hints,
            );

        let (min, max) = if matches!(I::BRANCHING, InputBranching::WithDeduplication { .. }) {
            (usize::from(self.seen.is_empty() && min > 0), max)
        } else {
            (min, max)
        };

        (min + self.todo.len(), max.map(|x| x + self.todo.len()))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Costs<W> {
    pub ignored: Option<W>,
    pub unmatched: Option<W>,
}

impl<W> Default for Costs<W> {
    fn default() -> Self {
        Costs {
            ignored: None,
            unmatched: None,
        }
    }
}

impl<W: ops::Mul<T>, T: Clone> ops::Mul<T> for Costs<W> {
    type Output = Costs<W::Output>;

    fn mul(self, rhs: T) -> Self::Output {
        Costs {
            ignored: self.ignored.map(|x| x * rhs.clone()),
            unmatched: self.unmatched.map(|x| x * rhs),
        }
    }
}

#[derive(Educe)]
#[educe(Clone(bound = "'static: 'a"))] // weird bound because of single use lifetime warning
pub enum ItemMatchInfo<'a, I: Input + ?Sized, PatItem> {
    Ignored {
        input: InputIterItem<I::State, I::Item>,
    },
    Unmatched {
        pattern: &'a PatItem,
        consume: bool,
        input_state: I::State,
    },
    Matched {
        pattern: &'a PatItem,
        consume: bool,
        input: InputIterItem<I::State, I::Item>,
    },
}

impl<I: Input + ?Sized, PatItem> Copy for ItemMatchInfo<'_, I, PatItem>
where
    I::State: Copy,
    I::Item: Copy,
{
}

impl<'a, I: Input + ?Sized, PatItem> ItemMatchInfo<'a, I, PatItem> {
    #[must_use]
    pub fn ignored(&self) -> bool {
        matches!(*self, ItemMatchInfo::Ignored { .. })
    }

    #[must_use]
    pub fn unmatched(&self) -> bool {
        matches!(*self, ItemMatchInfo::Unmatched { .. })
    }

    #[must_use]
    pub fn matched(&self) -> bool {
        matches!(*self, ItemMatchInfo::Matched { .. })
    }

    #[must_use]
    pub fn nonempty_input(&self) -> Option<&InputIterItem<I::State, I::Item>> {
        match *self {
            ItemMatchInfo::Ignored { ref input } | ItemMatchInfo::Matched { ref input, .. } => {
                Some(input)
            },
            ItemMatchInfo::Unmatched { .. } => None,
        }
    }

    #[must_use]
    pub fn input(&self) -> (ops::Range<&I::State>, Option<&I::Item>) {
        (
            self.input_range(),
            self.nonempty_input().map(|input| &input.item),
        )
    }

    #[must_use]
    pub fn pattern(&self) -> Option<(&'a PatItem, bool)> {
        match *self {
            ItemMatchInfo::Ignored { .. } => None,
            ItemMatchInfo::Unmatched {
                pattern, consume, ..
            }
            | ItemMatchInfo::Matched {
                pattern, consume, ..
            } => Some((pattern, consume)),
        }
    }

    #[must_use]
    pub fn pattern_item(&self) -> Option<&'a PatItem> {
        self.pattern().map(|(item, _)| item)
    }

    #[must_use]
    pub fn consume_input(&self) -> Option<bool> {
        self.pattern().map(|(_, consume)| consume)
    }

    #[must_use]
    pub fn input_state(&self) -> Either<(&I::State, &I::State), &I::State> {
        match *self {
            ItemMatchInfo::Ignored { ref input } | ItemMatchInfo::Matched { ref input, .. } => {
                Left((&input.before, &input.after))
            },
            ItemMatchInfo::Unmatched {
                ref input_state, ..
            } => Right(input_state),
        }
    }

    #[must_use]
    pub fn input_state_before(&self) -> &I::State {
        self.input_state()
            .either(|(before, _)| before, |state| state)
    }

    #[must_use]
    pub fn input_state_after(&self) -> &I::State {
        self.input_state().either(|(_, after)| after, |state| state)
    }

    #[must_use]
    pub fn input_range(&self) -> ops::Range<&I::State> {
        self.input_state()
            .either(|(before, after)| before..after, |state| state..state)
    }
}

impl<'a, LI, RI, AI, AL, AR, L, R> ItemMatchInfo<'a, MixedInput<LI, RI, AI>, Either<L, R>>
where
    LI: Input,
    RI: Input,
    AI: Input<Item = (AL, AR)>,
    AL: StateCheck<LI::State>,
    AR: StateCheck<RI::State>,
{
    #[must_use]
    pub fn mixed(self) -> Option<Either<ItemMatchInfo<'a, LI, L>, ItemMatchInfo<'a, RI, R>>> {
        match self {
            ItemMatchInfo::Ignored {
                input:
                    InputIterItem {
                        before,
                        item: Left(item),
                        after,
                    },
            } => Some(Left(ItemMatchInfo::Ignored {
                input: InputIterItem {
                    before: before.left,
                    item,
                    after: after.left,
                },
            })),
            ItemMatchInfo::Ignored {
                input:
                    InputIterItem {
                        before,
                        item: Right(item),
                        after,
                    },
            } => Some(Right(ItemMatchInfo::Ignored {
                input: InputIterItem {
                    before: before.right,
                    item,
                    after: after.right,
                },
            })),
            ItemMatchInfo::Unmatched {
                pattern: &Left(ref pattern),
                consume,
                input_state,
            } => Some(Left(ItemMatchInfo::Unmatched {
                pattern,
                consume,
                input_state: input_state.left,
            })),
            ItemMatchInfo::Unmatched {
                pattern: &Right(ref pattern),
                consume,
                input_state,
            } => Some(Right(ItemMatchInfo::Unmatched {
                pattern,
                consume,
                input_state: input_state.right,
            })),
            ItemMatchInfo::Matched {
                pattern: &Left(ref pattern),
                consume,
                input:
                    InputIterItem {
                        before,
                        item: Left(item),
                        after,
                    },
            } => Some(Left(ItemMatchInfo::Matched {
                pattern,
                consume,
                input: InputIterItem {
                    before: before.left,
                    item,
                    after: after.left,
                },
            })),
            ItemMatchInfo::Matched {
                pattern: &Right(ref pattern),
                consume,
                input:
                    InputIterItem {
                        before,
                        item: Right(item),
                        after,
                    },
            } => Some(Right(ItemMatchInfo::Matched {
                pattern,
                consume,
                input: InputIterItem {
                    before: before.right,
                    item,
                    after: after.right,
                },
            })),
            ItemMatchInfo::Matched {
                pattern: &Left(_),
                input: InputIterItem { item: Right(_), .. },
                ..
            }
            | ItemMatchInfo::Matched {
                pattern: &Right(_),
                input: InputIterItem { item: Left(_), .. },
                ..
            } => None,
        }
    }
}

fn mul_weight<W>(weight: &Option<W>, times: usize) -> Option<W>
where
    W: explore::Weight + ops::Mul<Output = W> + 'static,
    usize: AsPrimitive<W>,
{
    if times > 0 {
        Some(times.as_() * (*weight)?)
    } else {
        Some(W::default())
    }
}

pub trait TailImpl<I: Input, W: explore::Weight>: Sized + Clone {
    #[must_use]
    fn cost_tail(costs: &Costs<W>, input: I, state: I::State) -> Option<(W, Self)>;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct AnyTail;

impl<I: Input, W: explore::Weight> TailImpl<I, W> for AnyTail {
    fn cost_tail(_: &Costs<W>, _: I, _: <I as Input>::State) -> Option<(W, Self)> {
        Some((W::default(), AnyTail))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct IgnoreCostTail;

impl<I, W: 'static> TailImpl<I, W> for IgnoreCostTail
where
    I: Input,
    W: explore::Weight + ops::Mul<Output = W>,
    usize: AsPrimitive<W>,
{
    fn cost_tail(costs: &Costs<W>, input: I, state: I::State) -> Option<(W, Self)> {
        let ignored = match I::BRANCHING {
            InputBranching::None
            | InputBranching::NoDeduplication { const_depth: true }
            | InputBranching::WithDeduplication { const_depth: true } => {
                input.monotonic_iter(state).count()
            },
            InputBranching::NoDeduplication { const_depth: false }
            | InputBranching::WithDeduplication { const_depth: false } => input
                .branching_iter(state)
                .map(|(depth, _)| depth + 1)
                .max()
                .unwrap_or(0),
        };

        Some((mul_weight(&costs.ignored, ignored)?, IgnoreCostTail))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ReturnMonotonicTail<T>(pub T);

impl<T, I, W> TailImpl<I, W> for ReturnMonotonicTail<T>
where
    T: FromIterator<I::Item> + Clone,
    I: Input,
    W: explore::Weight,
{
    fn cost_tail(_: &Costs<W>, input: I, state: I::State) -> Option<(W, Self)> {
        Some((
            W::default(),
            ReturnMonotonicTail(input.monotonic_iter(state).map(|i| i.item).collect()),
        ))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct ReturnTailInputState<T>(pub T);

impl<I: Input, W: explore::Weight> TailImpl<I, W> for ReturnTailInputState<I::State> {
    fn cost_tail(_: &Costs<W>, _: I, state: I::State) -> Option<(W, Self)> {
        Some((W::default(), ReturnTailInputState(state)))
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Stats {
    pub max_match: usize,
    pub max_consume: usize,
}

impl Stats {
    fn record<Item>(&mut self, item: StatItem<Option<Item>>) {
        self.max_match += 1;
        if item.consume {
            self.max_consume += 1;
        }
    }

    fn join(self, right: Self) -> Self {
        Stats {
            max_match: self.max_match.max(right.max_match),
            max_consume: self.max_consume.max(right.max_consume),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash, Default)]
pub enum Advance {
    #[default]
    Advance,
    Stay,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, Hash)]
#[serde(try_from = "String", bound = "Matcher<Item>: FromStr<Err = String>")]
pub struct Matcher<Item>(Vec<Instruction<Item>>);

impl<Item> Matcher<Item> {
    #[must_use]
    pub const fn empty() -> Self {
        Matcher(Vec::new())
    }

    #[must_use]
    pub fn or(mut self, other: Self) -> Self {
        self.0.insert(0, Instruction::Skip {
            branch: true,
            count: self.0.len() + 1,
        });
        self.0.extend(
            iter::once(Instruction::Skip {
                branch: false,
                count: other.0.len(),
            })
            .chain(other.0),
        );
        self
    }

    pub fn parse(s: &str) -> Result<Self, String>
    where
        Self: FromStr<Err = String>,
    {
        Self::from_str(s)
    }

    fn stats_inner<S: Clone>(
        &self,
        initial: S,
        record: impl Fn(&mut S, StatItem<Item>),
        join: impl Fn(S, S) -> S,
    ) -> S {
        let mut branches_todo = Vec::new();
        let mut combined = None;

        let mut idx = 0;
        let mut current_stats = initial;
        loop {
            while idx < self.0.len() {
                match self.0[idx] {
                    Instruction::Item {
                        ref narrow,
                        consume,
                    } => {
                        record(&mut current_stats, StatItem { consume, narrow });
                        idx += 1;
                    },
                    Instruction::Skip { branch, count } => {
                        if branch {
                            branches_todo.push((idx + 1, current_stats.clone()));
                        }
                        idx += count + 1;
                    },
                }
            }

            let new_combined = match combined {
                None => current_stats,
                Some(combined) => join(combined, current_stats),
            };

            if let Some((new_idx, new_stats)) = branches_todo.pop() {
                combined = Some(new_combined);
                idx = new_idx;
                current_stats = new_stats;
                continue;
            } else {
                break new_combined;
            }
        }
    }

    pub fn explore_on<'a, I: 'a, D: 'a, S: 'a, O: 'a, W: 'a>(
        &'a self,
        input: I,
        input_from: Option<I::State>,
        initial: D,
        shared_static: S,
        advance: impl Fn(&S, &mut D, ItemMatchInfo<I, Item>) -> Option<W> + Clone + 'a,
        finalize: impl Fn(&S, D, InputMonotonicIter<I>) -> Option<(W, O)> + Clone + 'a,
    ) -> impl Explorer<W, (I::State, usize, D), Return = O> + 'a
    where
        I: Input + Clone,
        D: Clone + Hash,
        S: Clone,
        Item: Clone,
        W: explore::Weight,
        O: Clone,
    {
        #[coroutine] move |_| {
            let mut idx = 0;
            let mut input_state = input_from.unwrap_or_else(|| input.initial());
            let mut data = initial;
            while self.0.get(idx).is_some() {
                yield explore::Instruction::Dedup((input_state.clone(), idx, data.clone()));

                let instruction = match self.0[idx] {
                    Instruction::Item {
                        ref narrow,
                        consume,
                    } => Left((NoCopy(narrow), NoCopy(consume))),
                    Instruction::Skip { branch, count } => Right((NoCopy(branch), NoCopy(count))),
                };
                match instruction {
                    Left((pattern, consume)) => {
                        let (new_idx, match_info) = if explore::branch!(binary) {
                            let (next_state, item) = explore::branch!(
                                free: input.possible_nexts_with(input_state.clone())
                            );

                            let input = InputIterItem {
                                before: input_state,
                                item,
                                after: next_state.clone(),
                            };
                            input_state = next_state;

                            explore::branch!(free: [
                                (idx, ItemMatchInfo::Ignored {
                                    input: input.clone()
                                }),
                                (idx + 1, ItemMatchInfo::Matched {
                                    pattern: pattern.consume(),
                                    consume: consume.consume(),
                                    input
                                }),
                            ])
                        } else {
                            (idx + 1, ItemMatchInfo::Unmatched {
                                pattern: pattern.consume(),
                                consume: consume.consume(),
                                input_state: input_state.clone(),
                            })
                        };
                        idx = new_idx;
                        explore::try_push!(advance(&shared_static, &mut data, match_info));
                    },
                    Right((branch, count)) => {
                        if branch.consume() {
                            idx = explore::branch!(free: [idx + 1, idx + count.consume() + 1]);
                        } else {
                            idx += count.consume() + 1;
                        }
                    },
                }
            }

            match finalize(&shared_static, data, input.monotonic_iter(input_state))
                .map(|(w, o)| (NoCopy(w), o))
            {
                None => explore::fail!(),
                Some((weight, out)) => {
                    yield explore::Instruction::Push(weight.consume());
                    out
                },
            }
        }
    }

    #[allow(clippy::too_many_arguments)] // sufficiently typed
    pub fn trivially_explore_on<'a, I: 'a, D: 'a, W: 'a, T: 'a>(
        &'a self,
        input: I,
        input_from: Option<I::State>,
        initial: D,
        costs: Costs<W>,
        compare: impl Fn(&Item, &I::Item) -> Option<W> + Clone + 'a,
        assoc: impl Fn(&mut D, ItemMatchInfo<I, Item>) + Clone + 'a,
        finalize: impl Fn(&mut D, InputMonotonicIter<I>) + Clone + 'a,
    ) -> impl Explorer<W, (I::State, usize, D), Return = (D, T)> + 'a
    where
        I: Input + Clone,
        D: Clone + Hash,
        Item: Clone,
        W: explore::Weight,
        T: TailImpl<I, W>,
    {
        self.explore_on(
            input,
            input_from,
            initial,
            costs,
            move |costs, state, matched| {
                let cost = match matched {
                    ItemMatchInfo::Ignored { .. } => costs.ignored?,
                    ItemMatchInfo::Unmatched { .. } => costs.unmatched?,
                    ItemMatchInfo::Matched {
                        pattern, ref input, ..
                    } => compare(pattern, &input.item)?,
                };
                assoc(state, matched);
                Some(cost)
            },
            move |costs, mut data, rest| {
                let (cost, tail) =
                    T::cost_tail(costs, rest.input().clone(), rest.state_ref().clone())?;
                finalize(&mut data, rest);
                Some((cost, (data, tail)))
            },
        )
    }
}

impl<Item> Matcher<Option<Item>> {
    #[must_use]
    pub fn stats(&self) -> Stats {
        self.stats_inner(Stats::default(), Stats::record, Stats::join)
    }
}

impl<A, B> Matcher<Either<Option<A>, Option<B>>> {
    #[must_use]
    pub fn stats(&self) -> (Stats, Stats) {
        self.stats_inner(
            <(Stats, Stats)>::default(),
            |stats, item| match *item.narrow {
                Left(ref left) => stats.0.record(item.map_narrow(|_| left)),
                Right(ref right) => stats.1.record(item.map_narrow(|_| right)),
            },
            |left, right| (left.0.join(right.0), left.1.join(right.1)),
        )
    }
}

type MixedInputIters<'a, L, R, A> = (
    InputMonotonicIter<'a, L>,
    InputMonotonicIter<'a, R>,
    InputMonotonicIter<'a, MixedInput<L, R, A>>,
);

impl<L, R> Matcher<Either<L, R>> {
    #[allow(clippy::too_many_arguments)] // typing is sufficient to figure out what does what  //
    pub fn trivially_explore_on_mixed<'a, LI, RI, AI, AL, AR, D, W, TL, TR>(
        &'a self,
        input: MixedInput<LI, RI, AI>,
        input_from: Option<<MixedInput<LI, RI, AI> as Input>::State>,
        initial: D,
        costs_left: Costs<W>,
        costs_right: Costs<W>,
        compare_left: impl Fn(&L, &LI::Item) -> Option<W> + Clone + 'a,
        compare_right: impl Fn(&R, &RI::Item) -> Option<W> + Clone + 'a,
        assoc: impl Fn(&mut D, Either<ItemMatchInfo<LI, L>, ItemMatchInfo<RI, R>>) + Clone + 'a,
        finalize: impl Fn(&mut D, MixedInputIters<LI, RI, AI>) + Clone + 'a,
    ) -> impl Explorer<
        W,
        (<MixedInput<LI, RI, AI> as Input>::State, usize, D),
        Return = (D, TL, TR, MixedInputState<LI::State, RI::State, AI::State>),
    > + 'a
    where
        LI: Input + Clone + 'a,
        RI: Input + Clone + 'a,
        AI: Input<Item = (AL, AR)> + Clone + 'a,
        AL: StateCheck<LI::State>,
        AR: StateCheck<RI::State>,
        D: Clone + Hash + 'a,
        L: Clone,
        R: Clone,
        W: explore::Weight + 'a,
        TL: TailImpl<LI, W> + 'a,
        TR: TailImpl<RI, W> + 'a,
    {
        self.explore_on(
            input,
            input_from,
            initial,
            (costs_left, costs_right),
            move |costs, state, matched| {
                let matched = matched.clone().mixed()?;
                let cost = match matched {
                    Left(ItemMatchInfo::Ignored { .. }) => costs.0.ignored?,
                    Left(ItemMatchInfo::Unmatched { .. }) => costs.0.unmatched?,
                    Left(ItemMatchInfo::Matched {
                        pattern, ref input, ..
                    }) => compare_left(pattern, &input.item)?,
                    Right(ItemMatchInfo::Ignored { .. }) => costs.1.ignored?,
                    Right(ItemMatchInfo::Unmatched { .. }) => costs.1.unmatched?,
                    Right(ItemMatchInfo::Matched {
                        pattern, ref input, ..
                    }) => compare_right(pattern, &input.item)?,
                };
                assoc(state, matched);
                Some(cost)
            },
            move |costs, mut data, rest| {
                let MixedInput {
                    left: ref left_input,
                    right: ref right_input,
                    ..
                } = *rest.input();
                let state = rest.state_ref().clone();
                let (tail_cost_left, tail_left) =
                    TL::cost_tail(&costs.0, left_input.clone(), state.left.clone())?;
                let (tail_cost_right, tail_right) =
                    TR::cost_tail(&costs.1, right_input.clone(), state.right.clone())?;
                finalize(
                    &mut data,
                    (
                        left_input.monotonic_iter(state.left.clone()),
                        right_input.monotonic_iter(state.right.clone()),
                        rest,
                    ),
                );
                Some((
                    tail_cost_left + tail_cost_right,
                    (data, tail_left, tail_right, state),
                ))
            },
        )
    }
}

impl<'a, T> TryFrom<&'a str> for Matcher<T>
where
    Self: FromStr<Err = String>,
{
    type Error = String;

    fn try_from(value: &'a str) -> Result<Self, String> {
        Self::from_str(value)
    }
}

impl<T> TryFrom<String> for Matcher<T>
where
    Self: FromStr<Err = String>,
{
    type Error = String;

    fn try_from(value: String) -> Result<Self, String> {
        Self::from_str(&value)
    }
}

#[allow(clippy::module_name_repetitions)]
pub type OrthoMatcher = Matcher<Option<OrthoElement>>;

impl FromStr for OrthoMatcher {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        parser::parse(parser::context(parser::ortho_element), s)
    }
}

#[allow(clippy::module_name_repetitions)]
pub type IpaMatcher = Matcher<Option<IpaElement>>;

impl FromStr for IpaMatcher {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        parser::parse(parser::context(parser::ipa_element), s)
    }
}

#[allow(clippy::module_name_repetitions)]
pub type MixedMatcher = Matcher<Either<Option<IpaElement>, Option<OrthoElement>>>;

impl FromStr for MixedMatcher {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, String> {
        parser::parse(
            parser::alternate_context(parser::ipa_element, parser::ortho_element),
            s,
        )
    }
}

trait MixedInputAssertAssocMonotonic {
    const ASSERT: ();
}

pub trait StateCheck<T> {
    fn arrived(&self, at: &T) -> bool;
}

impl<T: PartialOrd> StateCheck<T> for T {
    fn arrived(&self, at: &T) -> bool {
        at >= self
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct NeverArrive;

impl<T> StateCheck<T> for NeverArrive {
    fn arrived(&self, _: &T) -> bool {
        false
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct AlwaysArrive;

impl<T> StateCheck<T> for AlwaysArrive {
    fn arrived(&self, _: &T) -> bool {
        true
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Debug)]
pub struct MixedInput<L, R, A> {
    pub left: L,
    pub right: R,
    pub assoc: A,
}

impl<L, R, A, AL, AR> MixedInput<L, R, A>
where
    L: Input,
    R: Input,
    A: Input<Item = (AL, AR)>,
    AL: StateCheck<L::State>,
    AR: StateCheck<R::State>,
{
    #[must_use]
    pub fn left_only(left: L) -> Self
    where
        R: Default,
        A: FromIterator<(NeverArrive, AlwaysArrive)>,
    {
        MixedInput {
            left,
            right: Default::default(),
            assoc: iter::once((NeverArrive, AlwaysArrive)).collect(),
        }
    }

    #[must_use]
    pub fn right_only(right: R) -> Self
    where
        L: Default,
        A: FromIterator<(AlwaysArrive, NeverArrive)>,
    {
        MixedInput {
            left: Default::default(),
            right,
            assoc: iter::once((AlwaysArrive, NeverArrive)).collect(),
        }
    }

    #[must_use]
    pub fn empty() -> Self
    where
        L: Default,
        R: Default,
        A: Default,
    {
        Self::default()
    }

    #[inline]
    fn translate_wide<F, AF, T>(
        &self,
        state: &F,
        from_initial: &F,
        extract_from: impl Fn(&(AL, AR)) -> &AF,
        extract_to: impl Fn((AL, AR)) -> T,
    ) -> (Option<T>, Option<T>)
    where
        AF: PartialEq + StateCheck<F>,
    {
        let mut assoc = self
            .assoc
            .monotonic_iter(self.assoc.initial())
            .map(|item| item.item)
            .skip_while(|item| extract_from(item).arrived(from_initial))
            .peekable();

        // follow the first which is equal on from to the last not arrived
        let mut last_changed = None;
        while let Some(item) = assoc.next_if(|item| extract_from(item).arrived(state)) {
            if last_changed
                .as_ref()
                .map_or(true, |old| extract_from(&item) != extract_from(old))
            {
                last_changed = Some(item);
            }
        }

        let lower = last_changed.map(&extract_to);

        // follow the last which is equal on from to the first arrived
        let upper = assoc.next().map(|item| {
            assoc
                .take_while(|next| extract_from(next) == extract_from(&item))
                .last()
                .map_or_else(|| extract_to(item), &extract_to)
        });

        (lower, upper)
    }

    #[inline]
    fn translate_narrow<F, AF, T>(
        &self,
        state: &F,
        extract_from: impl Fn(&(AL, AR)) -> &AF,
        extract_to: impl Fn((AL, AR)) -> T,
    ) -> (Option<T>, Option<T>)
    where
        AF: StateCheck<F>,
    {
        let mut assoc = self
            .assoc
            .monotonic_iter(self.assoc.initial())
            .map(|item| item.item)
            .peekable();

        // follow the last not arrived
        let mut lower = None;
        while let Some(item) = assoc.next_if(|item| extract_from(item).arrived(state)) {
            lower = Some(extract_to(item));
        }

        // follow the first arrived
        let upper = assoc.next().map(extract_to);

        (lower, upper)
    }

    // For `p#p#0a#1#2s#s` at `a` it returns `#0#1#2`; `None` = unbounded
    #[must_use]
    #[inline]
    pub fn left_to_right_wide(&self, state: &L::State) -> (Option<AR>, Option<AR>)
    where
        AL: PartialEq,
    {
        self.translate_wide(state, &self.left.initial(), |&(ref l, _)| l, |(_, r)| r)
    }

    // For `p#p#0a#1#2s#s` at `a` it returns `#1`; `None` = unbounded
    #[must_use]
    #[inline]
    pub fn left_to_right_narrow(&self, state: &L::State) -> (Option<AR>, Option<AR>) {
        self.translate_narrow(state, |&(ref l, _)| l, |(_, r)| r)
    }

    #[must_use]
    #[inline]
    pub fn right_to_left_wide(&self, state: &R::State) -> (Option<AL>, Option<AL>)
    where
        AR: PartialEq,
    {
        self.translate_wide(state, &self.right.initial(), |&(_, ref r)| r, |(l, _)| l)
    }

    #[must_use]
    #[inline]
    pub fn right_to_left_narrow(&self, state: &R::State) -> (Option<AL>, Option<AL>) {
        self.translate_narrow(state, |&(_, ref r)| r, |(l, _)| l)
    }

    fn make_range<F, T>(
        input: ops::Range<F>,
        make: impl Fn(F) -> (Option<T>, Option<T>),
    ) -> (ops::Bound<T>, ops::Bound<T>) {
        (
            make(input.start)
                .0
                .map_or(ops::Bound::Unbounded, ops::Bound::Included),
            make(input.end)
                .1
                .map_or(ops::Bound::Unbounded, ops::Bound::Excluded),
        )
    }

    #[must_use]
    #[inline]
    pub fn left_to_right_wide_range(
        &self,
        range: ops::Range<&L::State>,
    ) -> (ops::Bound<AR>, ops::Bound<AR>)
    where
        AL: PartialEq,
    {
        Self::make_range(range, |x| self.left_to_right_wide(x))
    }

    #[must_use]
    #[inline]
    pub fn left_to_right_narrow_range(
        &self,
        range: ops::Range<&L::State>,
    ) -> (ops::Bound<AR>, ops::Bound<AR>) {
        Self::make_range(range, |x| self.left_to_right_narrow(x))
    }

    #[must_use]
    #[inline]
    pub fn right_to_left_wide_range(
        &self,
        range: ops::Range<&R::State>,
    ) -> (ops::Bound<AL>, ops::Bound<AL>)
    where
        AR: PartialEq,
    {
        Self::make_range(range, |x| self.right_to_left_wide(x))
    }

    #[must_use]
    #[inline]
    pub fn right_to_left_narrow_range(
        &self,
        range: ops::Range<&R::State>,
    ) -> (ops::Bound<AL>, ops::Bound<AL>) {
        Self::make_range(range, |x| self.right_to_left_narrow(x))
    }
}

impl<L, R, A: Input> MixedInputAssertAssocMonotonic for MixedInput<L, R, A> {
    const ASSERT: () = assert!(
        matches!(A::BRANCHING, InputBranching::None),
        "Branching assocaitions are not supported as their implementation is very complex for \
         very little benefit."
    );
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct MixedInputState<L, R, A> {
    pub left: L,
    pub right: R,
    pub assoc: A,
}

impl<L, R, A, AL, AR> Input for MixedInput<L, R, A>
where
    L: Input,
    R: Input,
    A: Input<Item = (AL, AR)>,
    AL: StateCheck<L::State>,
    AR: StateCheck<R::State>,
{
    type Item = Either<L::Item, R::Item>;
    type State = MixedInputState<L::State, R::State, A::State>;

    const BRANCHING: InputBranching = InputBranching::WithDeduplication { const_depth: true };

    fn initial(&self) -> Self::State {
        MixedInputState {
            left: self.left.initial(),
            right: self.right.initial(),
            assoc: self.assoc.initial(),
        }
    }

    fn possible_final(&self) -> Self::State {
        MixedInputState {
            left: self.left.possible_final(),
            right: self.right.possible_final(),
            assoc: self.assoc.possible_final(),
        }
    }

    fn possible_nexts_with<'a>(
        &self,
        mut state: Self::State,
    ) -> impl Iterator<Item = (Self::State, Self::Item)> + Clone + 'a
    where
        Self: 'a,
    {
        #[allow(clippy::let_unit_value)]
        let () = Self::ASSERT; // compile time assert

        let iter_for =
            |state: Self::State, mut left: Option<_>, mut right: Option<_>| MixedInputNextIter {
                state,
                left_initial: left
                    .as_mut()
                    .and_then(|left: &mut (Option<_>, _)| left.0.take()),
                left: left.map(|left| left.1),
                right_initial: right
                    .as_mut()
                    .and_then(|right: &mut (Option<_>, _)| right.0.take()),
                right: right.map(|right| right.1),
            };

        while let Some((next_assoc, (left_bound, right_bound))) =
            self.assoc.possible_nexts_with(state.assoc.clone()).next()
        {
            let mut left_iter = Lazy::new(|| self.left.possible_nexts_with(state.left.clone()));
            let mut left_first = Lazy::new(|| left_iter.get_mut().next());
            let left_arrived = left_bound.arrived(&state.left) || left_first.get_mut().is_none();

            let mut right_iter = Lazy::new(|| self.right.possible_nexts_with(state.right.clone()));
            let mut right_first = Lazy::new(|| right_iter.get_mut().next());
            let right_arrived =
                right_bound.arrived(&state.right) || right_first.get_mut().is_none();

            if left_arrived && right_arrived {
                if left_first.get_mut().is_none() && right_first.get_mut().is_none() {
                    return iter_for(state, None, None);
                }

                state.assoc = next_assoc;
                continue;
            }

            let left = if left_arrived {
                None
            } else {
                Some((left_first.get(), left_iter.get()))
            };
            let right = if right_arrived {
                None
            } else {
                Some((right_first.get(), right_iter.get()))
            };

            return iter_for(state, left, right);
        }

        // `assoc` is done, so empty out the remaining sides
        MixedInputNextIter {
            left_initial: None,
            left: Some(self.left.possible_nexts_with(state.left.clone())),
            right_initial: None,
            right: Some(self.right.possible_nexts_with(state.right.clone())),
            state,
        }
    }

    fn monotonic_size_hint(&self, state: &Self::State) -> (usize, Option<usize>) {
        add_size_hints(
            self.left.monotonic_size_hint(&state.left),
            self.right.monotonic_size_hint(&state.right),
        )
    }
}

#[derive(Clone)]
pub struct MixedInputNextIter<LS, LI, LIt, RS, RI, RIt, AS> {
    state: MixedInputState<LS, RS, AS>,
    left_initial: Option<(LS, LI)>,
    left: Option<LIt>,
    right_initial: Option<(RS, RI)>,
    right: Option<RIt>,
}

impl<LS, LI, LIt, RS, RI, RIt, AS> Iterator for MixedInputNextIter<LS, LI, LIt, RS, RI, RIt, AS>
where
    LS: Clone,
    LIt: Iterator<Item = (LS, LI)>,
    RS: Clone,
    RIt: Iterator<Item = (RS, RI)>,
    AS: Clone,
{
    type Item = (MixedInputState<LS, RS, AS>, Either<LI, RI>);

    fn next(&mut self) -> Option<Self::Item> {
        let left_out = |(state, item)| {
            Some((
                MixedInputState {
                    left: state,
                    ..self.state.clone()
                },
                Left(item),
            ))
        };
        let right_out = |(state, item)| {
            Some((
                MixedInputState {
                    right: state,
                    ..self.state.clone()
                },
                Right(item),
            ))
        };

        if let Some(item) = self.left_initial.take() {
            return left_out(item);
        }

        if let Some(ref mut left) = self.left {
            match left.next() {
                Some(item) => return left_out(item),
                None => self.left = None,
            }
        }

        if let Some(item) = self.right_initial.take() {
            return right_out(item);
        }

        if let Some(ref mut right) = self.right {
            match right.next() {
                Some(item) => return right_out(item),
                None => self.right = None,
            }
        }

        None
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let const_len = |n| (n, Some(n));

        add_size_hints(
            add_size_hints(
                const_len(usize::from(self.left_initial.is_some())),
                self.left
                    .as_ref()
                    .map_or_else(|| const_len(0), Iterator::size_hint),
            ),
            add_size_hints(
                const_len(usize::from(self.right_initial.is_some())),
                self.right
                    .as_ref()
                    .map_or_else(|| const_len(0), Iterator::size_hint),
            ),
        )
    }
}
