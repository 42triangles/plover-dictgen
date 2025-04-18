use std::{
    cell::Cell,
    cmp::Reverse,
    collections::BinaryHeap,
    convert::Infallible,
    fmt,
    hash::Hash,
    iter,
    marker::Unpin,
    mem,
    ops::{self, Coroutine, CoroutineState},
    pin::Pin,
    rc::Rc,
};

use num_traits::AsPrimitive;
use rustc_hash::FxHashSet as HashSet;

use crate::helpers::WithOrdOf;

#[derive(Clone, Default)]
pub enum Instruction<W, D> {
    #[default]
    Fail,
    Push(W),
    Branch,
    Dedup(D),
    Heuristic(W),
}

impl<W, D> Instruction<W, D> {
    fn map_dedup<D2>(self, dedup: impl Fn(D) -> D2) -> Instruction<W, D2> {
        match self {
            Instruction::Fail => Instruction::Fail,
            Instruction::Push(w) => Instruction::Push(w),
            Instruction::Branch => Instruction::Branch,
            Instruction::Dedup(x) => Instruction::Dedup(dedup(x)),
            Instruction::Heuristic(w) => Instruction::Heuristic(w),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub enum BranchTaken {
    #[default]
    Left,
    Right,
}

#[derive(Clone, Copy, Default)]
pub struct State<W> {
    pub weight: W,
    pub branch: BranchTaken,
}

fn map_instruction<W: Weight, D: Clone, D2: Clone, O: Clone>(
    mut e: impl Explorer<W, D, Return = O>,
    f: impl Fn(Instruction<W, D>) -> Option<Instruction<W, D2>> + Clone,
) -> impl Explorer<W, D2, Return = O> {
    #[coroutine] move |mut state| loop {
        match Pin::new(&mut e).resume(state) {
            CoroutineState::Yielded(instruction) => match f(instruction) {
                None => continue,
                Some(instruction) => state = yield instruction,
            },
            CoroutineState::Complete(out) => break out,
        }
    }
}

#[must_use]
pub trait Explorer<W: Weight, D: Clone>:
    Coroutine<State<W>, Yield = Instruction<W, D>> + Clone + Unpin
{
    fn map_dedup<D2: Clone>(
        self,
        f: impl Fn(D) -> D2 + Clone,
    ) -> impl Explorer<W, D2, Return = Self::Return>
    where
        Self::Return: Clone,
    {
        map_instruction(self, move |i| Some(i.map_dedup(&f)))
    }

    fn without_heuristic(self) -> impl Explorer<W, D, Return = Self::Return>
    where
        Self::Return: Clone,
    {
        map_instruction(self, |i| match i {
            Instruction::Heuristic(_) => None,
            instruction => Some(instruction),
        })
    }

    fn add_heuristic_after(self, add: W) -> impl Explorer<W, D, Return = Self::Return>
    where
        Self::Return: Clone,
    {
        map_instruction(self, move |i| {
            Some(match i {
                Instruction::Heuristic(h) => Instruction::Heuristic(h + add),
                instruction => instruction,
            })
        })
    }

    fn without_dedup<D2: Clone>(self) -> impl Explorer<W, D2, Return = Self::Return>
    where
        Self::Return: Clone,
    {
        map_instruction(self, |i| match i {
            Instruction::Dedup(_) => None,
            instruction => Some(instruction.map_dedup(|_| unreachable!())),
        })
    }
}

impl<W: Weight, D: Clone, T> Explorer<W, D> for T where
    T: Coroutine<State<W>, Yield = Instruction<W, D>> + Clone + Unpin
{
}

#[macro_export]
macro_rules! fail {
    () => {{
        yield $crate::explore::Instruction::Fail;
        unreachable!();
    }};
}

pub use crate::fail;

#[macro_export]
macro_rules! try_push {
    ($weight:expr $(,)?) => {{
        let instruction = match $weight {
            None => Some($crate::explore::Instruction::Fail),
            Some(w) => Some($crate::explore::Instruction::Push(w)),
        };
        if let Some(instruction) = instruction {
            yield instruction;
        }
    }};
}

pub use crate::try_push;

// NOTE: Returning an `impl Iterator` is necessary due to https://github.com/rust-lang/rust/issues/118153
#[doc(hidden)]
pub fn free_branch_proto<T, It>(branches: It) -> impl Iterator<Item = T> + Clone
where
    T: Clone,
    It: IntoIterator<Item = T>,
    It::IntoIter: Clone,
{
    branches.into_iter()
}

#[macro_export]
macro_rules! branch {
    (binary) => {{
        let state: $crate::explore::State<_> = yield $crate::explore::Instruction::Branch;

        state.branch == $crate::explore::BranchTaken::Left
    }};
    (free: [] $(,)?) => {
        $crate::explore::fail!()
    };
    (free: [$elem:expr $(,)?] $(,)?) => {{
        $elem
    }};
    (free: [$first:expr $(, $elem:expr)+ $(,)?] $(,)?) => {{
        if branch!(binary) {
            $first
        } else {
            $crate::explore::branch!(free: [$($elem),+])
        }
    }};
    (free: $from:tt .. $to:tt $(,)?) => {{
        if $from == $to {
            $crate::explore::fail!();
        }

        let mut out = $from;

        while out + 1 < $to {
            if explore::branch!(binary) {
                break;
            }
            out += 1;
        }

        out
    }};
    (free: $iter:expr $(,)?) => {{
        let mut iter = $crate::explore::free_branch_proto($iter);

        let Some(mut last) = iter.next() else {
            $crate::explore::fail!();
        };

        for next in iter {
            if explore::branch!(binary) {
                break;
            }
            last = next;
        }

        last
    }};
    (weighted: $iter:expr $(,)?) => {{
        let (weight, out) = {
            let (weight, out) = $crate::explore::branch!(free: $iter);
            ($crate::helpers::NoCopy(weight), out)
        };

        yield $crate::explore::Instruction::Push(weight.consume());

        out
    }};
}

pub use crate::branch;

#[doc(hidden)]
pub fn coerce_explorer<W: Weight, D: Clone, E: Explorer<W, D>>(x: E) -> E {
    x
}

#[doc(hidden)]
pub fn coerce_same_type_as<T>(_: &T, out: T) -> T {
    out
}

#[macro_export]
macro_rules! call {
    (@staggered $call:expr, $stagger:expr, $branch:expr, |$w:ident| $max:expr) => {{
        let iter_cell = std::rc::Rc::new(
            std::cell::Cell::new(Some($crate::explore::staggered_run($stagger, $call)))
        );
        let mut last_weight = Default::default();

        loop {
            let next = {
                let mut iter = iter_cell.take().unwrap();
                let next = iter.next().filter(|&($w, _)| $max);
                iter_cell.set(Some(iter));
                next
            };
            match next {
                None => {
                    yield $crate::explore::Instruction::Fail;
                },
                Some((w, out)) => {
                    $crate::explore::coerce_same_type_as(&w, last_weight);

                    let diff = w - last_weight;
                    last_weight = w;

                    yield $crate::explore::Instruction::Push(diff);

                    if let Some(out) = out {
                        if $branch {
                            break out;
                        }
                    }
                },
            };
        }
    }};
    ($call:expr, staggered $stagger:expr $(,)?) => {
        $crate::explore::call!(@staggered $call, $stagger, $crate::explore::branch!(binary), |_| true)
    };
    ($call:expr, max $max:expr $(,)?) => {
        $crate::explore::call!(@staggered $call, $max, $crate::explore::branch!(binary), |w| w <= $max)
    };
    ($call:expr, staggered $stagger:expr, best $(,)?) => {
        $crate::explore::call!(@staggered $call, $stagger, true, |_| true)
    };
    ($call:expr, staggered $stagger:expr, restart, best $(,)?) => {{
        let mut explorer = Some($call);
        let mut last_weight = Default::default();

        loop {
            let result = $crate::helpers::NoCopy(
                $crate::explore::staggered_run(last_weight + $stagger, explorer.clone().unwrap())
                    .next()
            );

            match result.consume() {
                None => {
                    yield $crate::explore::Instruction::Fail;
                },
                Some((w, out)) => {
                    $crate::explore::coerce_same_type_as(&w, last_weight);

                    let diff = w - last_weight;
                    last_weight = w;

                    if out.is_some() {
                        ::std::mem::drop(explorer.take());
                    }

                    yield $crate::explore::Instruction::Push(diff);

                    if let Some(out) = out {
                        break out;
                    }
                }
            }
        }
    }};
    ($call:expr, best $(,)?) => {{
        let Some((weight, out)) = $crate::explore::run($call).next()
            .map(|(w, o)| ($crate::helpers::NoCopy(w), o))
        else {
            $crate::explore::fail!();
        };

        yield $crate::explore::Instruction::Push(weight.consume());

        out
    }};
    ($call:expr, dedup: |$x:ident| $dedup:expr, heuristic: $h:expr $(,)?) => {{
        use std::ops::{Coroutine, CoroutineState};
        use $crate::explore::{coerce_explorer, coerce_same_type_as, Instruction, State};

        let mut inner = coerce_explorer($call);
        let mut state = State::default();
        loop {
            match std::pin::Pin::new(&mut inner).resume(state) {
                CoroutineState::Yielded(instruction) => state = yield match instruction {
                    Instruction::Fail => Instruction::Fail,
                    Instruction::Push(w) => Instruction::Push(w),
                    Instruction::Branch => Instruction::Branch,
                    Instruction::Dedup($x) => Instruction::Dedup($dedup),
                    Instruction::Heuristic(w) => Instruction::Heuristic(w + coerce_same_type_as(&w, $h)),
                },
                CoroutineState::Complete(out) => break out,
            }
        }
    }};
    ($call:expr, dedup: |$x: ident| $dedup:expr $(,)?) => {
        $crate::call!($call, dedup: |$x| $dedup, heuristic: Default::default())
    };
    ($call:expr, heuristic: $h:expr $(,)?) => {
        $crate::call!($call, dedup: |x| x, heuristic: $h)
    };
    ($call:expr $(,)?) => {
        $crate::call!($call, dedup: |x| x, heuristic: Default::default())
    };
}

pub use crate::call;

pub trait Weight: Copy + Ord + Default + ops::Add<Output = Self> + fmt::Debug {
    fn update(&mut self, actual_weight: &mut Self, additional: Self);
    fn update_heuristic(&mut self, actual_weight: &mut Self, new_heuristic: Self);
    fn cutoff(iter: impl IntoIterator<Item = Self>) -> Self;
}

impl<T> Weight for T
where
    T: Copy
        + Ord
        + Default
        + ops::Add<Output = Self>
        + ops::Div<Output = Self>
        + fmt::Debug
        + 'static,
    usize: AsPrimitive<T>,
{
    fn update(&mut self, actual_weight: &mut Self, additional: Self) {
        *actual_weight = *actual_weight + additional;
        if *actual_weight > *self {
            *self = *actual_weight;
        }
    }

    fn update_heuristic(&mut self, actual_weight: &mut Self, new_heuristic: Self) {
        //debug_assert!(*actual_weight + new_heuristic >= *self);
        *self = *actual_weight + new_heuristic;
    }

    fn cutoff(iter: impl IntoIterator<Item = Self>) -> Self {
        let mut sum = 0.as_();
        let mut count = 0;
        for i in iter {
            sum = sum + i;
            count += 1;
        }

        sum / (count + 1).as_()
    }
}

#[allow(clippy::use_debug, clippy::too_many_lines)]
pub fn controlled_run<'a, W, D, O, E, B>(
    explorer: E,
    cut_after: Option<usize>,
    check_weight: impl Fn(W) -> Result<Option<B>, B> + 'a,
) -> impl Iterator<Item = Result<(W, O), B>> + 'a
where
    W: Weight + 'a,
    D: Clone + Hash + Eq + 'a,
    E: Explorer<W, D, Return = O> + 'a,
{
    let mut seen = HashSet::default();
    let mut zero_queue = Vec::new();
    let mut heap_todo = Vec::new();
    let mut heap = BinaryHeap::<WithOrdOf<Reverse<_>, _>>::new();
    let mut last_weight = W::default();
    let mut max_weight = None;

    zero_queue.push(WithOrdOf(
        Reverse(W::default()),
        (W::default(), Box::new(explorer)),
    ));

    iter::from_fn(move || 'outer: loop {
        let (mut item, mut branch) = match zero_queue.pop() {
            None => {
                if !heap_todo.is_empty() {
                    #[allow(clippy::iter_with_drain)] // we do want to keep the capacity around
                    heap.extend(heap_todo.drain(..));

                    if let Some(max_heap) = cut_after {
                        if heap.len() > max_heap {
                            let len_before = heap.len();

                            while heap.len() > 2 * len_before / 3 {
                                let cutoff = W::cutoff(heap.iter().map(|i| i.0 .0));
                                max_weight = Some(cutoff);
                                heap.retain(|i| i.0 .0 < cutoff);
                            }

                            #[allow(clippy::cast_precision_loss)]
                            {
                                eprintln!(
                                    "Cut exploration to {:.1}%",
                                    heap.len() as f32 * 100. / len_before as f32
                                );
                            }
                        }
                    }
                }

                let Some(item) = heap.pop() else {
                    mem::take(&mut seen);
                    mem::take(&mut zero_queue);
                    mem::take(&mut heap_todo);
                    mem::take(&mut heap);

                    return None;
                };

                if item.0 .0 > last_weight {
                    last_weight = item.0 .0;

                    match check_weight(item.0 .0) {
                        Ok(None) => (),
                        Ok(Some(out)) => {
                            zero_queue.push(item);
                            return Some(Err(out));
                        },
                        Err(out) => {
                            mem::take(&mut seen);
                            mem::take(&mut zero_queue);
                            mem::take(&mut heap_todo);
                            mem::take(&mut heap);

                            return Some(Err(out));
                        },
                    }
                }

                (item, BranchTaken::Left)
            },
            Some(out) => (out, BranchTaken::Right),
        };

        loop {
            match Pin::new(&mut item.1 .1).resume(State {
                weight: item.1 .0,
                branch,
            }) {
                CoroutineState::Yielded(Instruction::Fail) => continue 'outer,
                CoroutineState::Yielded(Instruction::Push(extra)) if extra == W::default() => {
                    continue
                },
                CoroutineState::Yielded(Instruction::Push(extra)) => {
                    item.0 .0.update(&mut item.1 .0, extra);
                    if max_weight.map_or(true, |max| item.0 .0 <= max) {
                        heap_todo.push(item);
                    }
                    continue 'outer;
                },
                CoroutineState::Yielded(Instruction::Branch) => {
                    zero_queue.push(WithOrdOf(item.0, (item.1 .0, item.1 .1.clone())));
                    branch = BranchTaken::Left;
                },
                CoroutineState::Yielded(Instruction::Dedup(marker)) => {
                    if seen.insert(marker) {
                        continue;
                    } else {
                        continue 'outer;
                    }
                },
                CoroutineState::Yielded(Instruction::Heuristic(heuristic)) => {
                    let last_weight = item.0 .0;
                    item.0 .0.update_heuristic(&mut item.1 .0, heuristic);
                    if last_weight == item.0 .0 {
                        continue;
                    } else {
                        heap_todo.push(item);
                        continue 'outer;
                    }
                },
                CoroutineState::Complete(out) => {
                    if item.1 .0 < item.0 .0 {
                        eprintln!(
                            "Heuristic ({:?}) was bigger than actual weight ({:?})",
                            item.0 .0, item.1 .0
                        );
                    }

                    return Some(Ok((item.1 .0, out)));
                },
            }
        }
    })
}

pub fn run<'a, W, D, O, E>(
    explorer: E,
    cut_after: Option<usize>,
) -> impl Iterator<Item = (W, O)> + 'a
where
    W: Weight + 'a,
    D: Clone + Hash + Eq + 'a,
    E: Explorer<W, D, Return = O> + 'a,
{
    controlled_run(explorer, cut_after, |_| Ok::<_, Infallible>(None)).map(|x| match x {
        Ok(out) => out,
        Err(infallible) => match infallible {},
    })
}

pub fn staggered_run<'a, W, D, O, E>(
    stagger: W,
    explorer: E,
    cut_after: Option<usize>,
) -> impl Iterator<Item = (W, Option<O>)> + 'a
where
    W: Weight + 'a,
    D: Clone + Hash + Eq + 'a,
    E: Explorer<W, D, Return = O> + 'a,
{
    let next_weight = Rc::new(Cell::new(W::default() + stagger));
    let next_weight_clone = Rc::clone(&next_weight);
    controlled_run(explorer, cut_after, move |w| {
        if w < next_weight.get() {
            Ok(None)
        } else {
            next_weight.set(w + stagger);
            Ok(Some(w))
        }
    })
    .map(move |item| match item {
        Ok((weight, out)) => {
            next_weight_clone.set(weight + stagger);
            (weight, Some(out))
        },
        Err(weight) => (weight, None),
    })
}
