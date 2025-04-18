use std::{
    cell::Cell,
    cmp::Reverse,
    collections::{hash_map::Entry, BinaryHeap},
    fmt, fs, iter, mem, ops,
    path::{Path, PathBuf},
    rc::Rc,
};

use educe::Educe;
use either::{Either, Left, Right};
use itertools::Itertools;
use rustc_hash::FxHashMap as HashMap;
use serde::{de::DeserializeOwned, Deserialize};
use typed_arena::Arena;

use crate::{
    arpabet,
    explore::{self, Explorer},
    helpers::{
        DeserializeVia, HashRefList, NoCopy, PtrOrd, RefList, SmallBitSet, TomlOption,
        UntaggedEither, WithOrdOf,
    },
    ipa::{self, StressMarker},
    language::Language,
    matcher::{self, ItemMatchInfo, MixedInputState, MixedMatcher},
    IpaWeights, OrthoWeights, Weight,
};

type MixedInput<'a> = matcher::MixedInput<&'a [ipa::Element], &'a str, &'a [(usize, usize)]>;

#[derive(Debug)]
pub struct PreparedWord {
    pub ipa: Vec<ipa::Element>,
    pub ortho: String,
    pub stress_markers: Vec<(usize, StressMarker)>,
    pub ortho_mapping: Vec<(usize, usize)>,
}

impl fmt::Display for PreparedWord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mapping = iter::once(&(0, 0))
            .chain(&self.ortho_mapping)
            .zip(&self.ortho_mapping);
        for (&from, &to) in mapping {
            for i in from.0..to.0 {
                let marker = self.stress_markers.iter().find(|&&(idx, _)| idx == i);
                if let Some(&(_, marker)) = marker {
                    match marker {
                        StressMarker::None => write!(f, ".")?,
                        StressMarker::Secondary => write!(f, ",")?,
                        StressMarker::Primary => write!(f, "'")?,
                    }
                }

                write!(f, "{}", self.ipa[i])?;
            }

            let ortho = &self.ortho[from.1..to.1];
            match ortho.chars().count() {
                0 => write!(f, "##")?,
                1 => write!(f, "#{ortho}")?,
                _ => write!(f, "#({ortho})")?,
            }
        }

        Ok(())
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Default)]
pub struct ChordString {
    pub string: String,
    pub char_count: usize,
}

impl ChordString {
    #[must_use]
    pub fn new(string: String) -> Self {
        ChordString {
            char_count: string.chars().count(),
            string,
        }
    }
}

impl<'de> Deserialize<'de> for ChordString {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self::new(String::deserialize(deserializer)?))
    }
}

impl fmt::Debug for ChordString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.string)
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
struct Chord(SmallBitSet);

impl Chord {
    fn overlaps_with(&self, rhs: &Self) -> bool {
        self.0.intersects_with(&rhs.0)
    }

    fn ordered_set_bits<'a>(
        &'a self,
        unordered: &'a Self,
    ) -> impl DoubleEndedIterator<Item = usize> + 'a {
        (0..self.0.capacity()).filter(move |&i| self.0.bit(i) && !unordered.0.bit(i))
    }

    fn starts_after(&self, rhs: &Self, unordered: &Self) -> bool {
        match (
            self.ordered_set_bits(unordered).next(),
            rhs.ordered_set_bits(unordered).next_back(),
        ) {
            (Some(l), Some(r)) if l > r => true,
            (Some(_), Some(_)) => false,
            _ => true,
        }
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    fn extend(&mut self, with: &Self) {
        self.0 |= &with.0;
    }
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct Layout {
    order: ChordString,
    #[serde(default)]
    hyphen_before: TomlOption<char>,
    #[serde(default)]
    unordered: ChordString,
    #[serde(default)]
    disambiguators: ChordString,
}

impl Layout {
    fn parse_str(&self, mut s: &str) -> Chord {
        let mut out = self.empty_chord();

        for (idx, i) in self.order.string.chars().enumerate() {
            if let Some(rest) = s
                .strip_prefix('-')
                .filter(|_| Some(i) == self.hyphen_before.0)
            {
                s = rest;
            }

            if let Some(rest) = s.strip_prefix(i) {
                out.0.set(idx);
                s = rest;
            }
        }

        out
    }

    fn parse(&self, s: &ChordString) -> Chord {
        self.parse_str(&s.string)
    }

    fn render(&self, s: &Chord) -> ChordString {
        let mut out = ChordString::default();

        let mut hyphen_needed = false;
        for (idx, i) in self.order.string.chars().enumerate() {
            if Some(i) == self.hyphen_before.0 {
                hyphen_needed = true;
            }

            if s.0.bit(idx) {
                if hyphen_needed {
                    out.string.push('-');
                    out.char_count += 1;
                    hyphen_needed = false;
                }
                out.string.push(i);
                out.char_count += 1;
            }
        }

        out
    }

    fn empty_chord(&self) -> Chord {
        Chord(SmallBitSet::with_capacity(self.order.char_count))
    }
}

#[derive(Clone, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct UnlessInner {
    matching: MixedMatcher,
    importance: Option<Weight>,
    threshold: Weight,
}

#[allow(clippy::fallible_impl_from)] // 1.0 isn't NaN
impl From<UntaggedEither<MixedMatcher, UnlessInner>> for UnlessInner {
    fn from(value: UntaggedEither<MixedMatcher, UnlessInner>) -> Self {
        match value {
            UntaggedEither::Left(matching) => UnlessInner {
                matching,
                importance: None,
                threshold: Weight::new(0.5).unwrap(),
            },
            UntaggedEither::Right(unless) => unless,
        }
    }
}

type Unless = DeserializeVia<UntaggedEither<MixedMatcher, UnlessInner>, UnlessInner>;

fn return_false() -> bool {
    false
}

#[derive(Deserialize, Default, Debug)]
#[serde(rename_all = "kebab-case")]
struct RuleInner {
    to: ChordString,
    #[serde(default)]
    alternative: Option<UntaggedEither<ChordString, Vec<ChordString>>>,
    #[serde(default)]
    unless: Option<UntaggedEither<Unless, Vec<Unless>>>,
    #[serde(default)]
    excludes: Vec<ChordString>,
    #[serde(default)]
    add_weight: Weight,
    #[serde(default = "return_false")]
    unordered: bool,
    #[serde(default = "return_false")]
    syllable_lookahead: bool,
}

impl From<UntaggedEither<ChordString, RuleInner>> for RuleInner {
    fn from(value: UntaggedEither<ChordString, RuleInner>) -> Self {
        match value {
            UntaggedEither::Left(to) => RuleInner {
                to,
                ..RuleInner::default()
            },
            UntaggedEither::Right(rule) => rule,
        }
    }
}

type Rule = DeserializeVia<UntaggedEither<ChordString, RuleInner>, RuleInner>;

#[derive(Debug)]
struct Alternative {
    to: Chord,
    blocked: Chord,
}

#[derive(Debug)]
struct CompiledRule {
    from: MixedMatcher,
    main: Alternative,
    alternatives: Vec<Alternative>,
    unless: Vec<UnlessInner>,
    excludes: Vec<(usize, usize)>,
    excludes_chords: Vec<Chord>,
    add_weight: Weight,
    unordered: bool,
    syllable_lookahead: bool,
    force_next_group: bool,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
#[allow(clippy::struct_field_names)]
struct Rules {
    order: Vec<String>,
    #[serde(default)]
    max_one: Vec<String>,
    #[serde(flatten)]
    rules: HashMap<String, HashMap<MixedMatcher, UntaggedEither<Rule, Vec<Rule>>>>,
}

impl Rules {
    fn compile(&self, layout: &Layout, unordered: &Chord) -> Vec<Vec<CompiledRule>> {
        let alternative = |to: &ChordString| {
            let to = layout.parse(to);

            let mut blocked = to.clone();
            if let (Some(first), Some(last)) = (
                to.ordered_set_bits(unordered).next(),
                to.ordered_set_bits(unordered).next_back(),
            ) {
                for i in first..=last {
                    blocked.0.set(i);
                }
            }

            Alternative { to, blocked }
        };

        let mut out = self
            .order
            .iter()
            .map(|i| {
                let force_next_group = self.max_one.contains(i);

                self.rules[i]
                    .iter()
                    .flat_map(|(from, rules)| match *rules {
                        UntaggedEither::Left(ref single) => Left(iter::once((from, single))),
                        UntaggedEither::Right(ref many) => {
                            Right(many.iter().map(move |rule| (from, rule)))
                        },
                    })
                    .map(|(from, rule)| CompiledRule {
                        from: from.clone(),
                        main: alternative(&rule.0.to),
                        alternatives: match rule.0.alternative {
                            None => vec![],
                            Some(UntaggedEither::Left(ref single)) => vec![alternative(single)],
                            Some(UntaggedEither::Right(ref many)) => {
                                many.iter().map(alternative).collect()
                            },
                        },
                        unless: match rule.0.unless {
                            None => vec![],
                            Some(UntaggedEither::Left(ref single)) => vec![single.0.clone()],
                            Some(UntaggedEither::Right(ref many)) => {
                                many.iter().map(|i| i.0.clone()).collect()
                            },
                        },
                        excludes: Vec::new(),
                        excludes_chords: rule.0.excludes.iter().map(|i| layout.parse(i)).collect(),
                        add_weight: rule.0.add_weight,
                        unordered: rule.0.unordered,
                        syllable_lookahead: rule.0.syllable_lookahead,
                        force_next_group,
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let excludes = {
            let out = &out;

            out.iter()
                .enumerate()
                .flat_map(|(idx, i)| i.iter().enumerate().map(move |x| (idx, x)))
                .flat_map(|(group_l, (rule_l, i))| {
                    i.excludes_chords.iter().flat_map(move |to| {
                        out.iter().enumerate().flat_map(move |(group_r, i)| {
                            i.iter()
                                .enumerate()
                                .filter(|&(_, rule)| rule.main.to == *to)
                                .map(move |(rule_r, _)| ((group_l, rule_l), (group_r, rule_r)))
                        })
                    })
                })
                .collect::<Vec<_>>()
        };

        for (l, r) in excludes {
            let mut write = |subject: (usize, usize), excluded| {
                let subject = &mut out[subject.0][subject.1].excludes;
                if !subject.contains(&excluded) {
                    subject.push(excluded);
                }
            };
            write(l, r);
            write(r, l);
        }

        /*
        for (group_idx, group_name) in self.order.iter().enumerate() {
            for (rule_idx, rule) in self.rules[&*group_name].iter().enumerate() {
                println!("{group_idx} {rule_idx}: {:?} {:?}", rule.1 .0.to, out[group_idx][rule_idx].to);
            }
        }
        */

        out
    }
}

#[derive(Educe, Debug)]
#[educe(Default(expression = "Self(vec![TomlOption(None)])"))]
struct OrderViolationWeightsInner(Vec<TomlOption<Weight>>);

impl From<UntaggedEither<TomlOption<Weight>, Vec<TomlOption<Weight>>>>
    for OrderViolationWeightsInner
{
    fn from(value: UntaggedEither<TomlOption<Weight>, Vec<TomlOption<Weight>>>) -> Self {
        match value {
            UntaggedEither::Left(single_item) => OrderViolationWeightsInner(vec![single_item]),
            UntaggedEither::Right(items) => OrderViolationWeightsInner(items),
        }
    }
}

type OrderViolationWeights = DeserializeVia<
    UntaggedEither<TomlOption<Weight>, Vec<TomlOption<Weight>>>,
    OrderViolationWeightsInner,
>;

#[derive(Clone, Default, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct JoinDropWeightsDirectionalInner {
    left: TomlOption<Weight>,
    right: TomlOption<Weight>,
}

impl From<UntaggedEither<TomlOption<Weight>, JoinDropWeightsDirectionalInner>>
    for JoinDropWeightsDirectionalInner
{
    fn from(value: UntaggedEither<TomlOption<Weight>, JoinDropWeightsDirectionalInner>) -> Self {
        match value {
            UntaggedEither::Left(weight) => JoinDropWeightsDirectionalInner {
                left: weight,
                right: weight,
            },
            UntaggedEither::Right(weights) => weights,
        }
    }
}

type JoinDropWeightsDirectional = DeserializeVia<
    UntaggedEither<TomlOption<Weight>, JoinDropWeightsDirectionalInner>,
    JoinDropWeightsDirectionalInner,
>;

#[derive(Default, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct JoinDropWeightsInner {
    #[serde(default)]
    primary: JoinDropWeightsDirectional,
    #[serde(default)]
    secondary: JoinDropWeightsDirectional,
    none: JoinDropWeightsDirectional,
}

impl From<UntaggedEither<JoinDropWeightsDirectional, JoinDropWeightsInner>>
    for JoinDropWeightsInner
{
    fn from(value: UntaggedEither<JoinDropWeightsDirectional, JoinDropWeightsInner>) -> Self {
        match value {
            UntaggedEither::Left(directional) => JoinDropWeightsInner {
                primary: directional.clone(),
                secondary: directional.clone(),
                none: directional,
            },
            UntaggedEither::Right(weights) => weights,
        }
    }
}

type JoinDropWeights = DeserializeVia<
    UntaggedEither<JoinDropWeightsDirectional, JoinDropWeightsInner>,
    JoinDropWeightsInner,
>;

#[derive(Default, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct WeightsByStressInner {
    #[serde(default)]
    primary: TomlOption<Weight>,
    #[serde(default)]
    secondary: TomlOption<Weight>,
    none: TomlOption<Weight>,
}

impl From<UntaggedEither<TomlOption<Weight>, WeightsByStressInner>> for WeightsByStressInner {
    fn from(value: UntaggedEither<TomlOption<Weight>, WeightsByStressInner>) -> Self {
        match value {
            UntaggedEither::Left(weight) => WeightsByStressInner {
                primary: weight,
                secondary: weight,
                none: weight,
            },
            UntaggedEither::Right(weights) => weights,
        }
    }
}

type WeightsByStress =
    DeserializeVia<UntaggedEither<TomlOption<Weight>, WeightsByStressInner>, WeightsByStressInner>;

#[derive(Default, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct SyllableWeights {
    #[serde(default)]
    join: TomlOption<Weight>,
    #[serde(default)]
    join_drop: JoinDropWeights,
    #[serde(default)]
    split_between: TomlOption<Weight>,
    #[serde(default)]
    split_inside: TomlOption<Weight>,
    #[serde(default)]
    split_inside_start: Option<String>,
    #[serde(default)]
    split_lookahead: TomlOption<Weight>,
    #[serde(default)]
    drop_fully: WeightsByStress,
    #[serde(default)]
    split_at_nucleus: WeightsByStress,
}

fn neutral_factor() -> Weight {
    Weight::new(1.0).unwrap()
}

#[derive(Deserialize, Educe, Debug)]
#[educe(Default)]
#[serde(rename_all = "kebab-case")]
struct StressAdjustItemInner {
    #[educe(Default(expression = "neutral_factor()"))]
    #[serde(default = "neutral_factor")]
    onset: Weight,
    #[educe(Default(expression = "neutral_factor()"))]
    #[serde(default = "neutral_factor")]
    nucleus: Weight,
    #[educe(Default(expression = "neutral_factor()"))]
    #[serde(default = "neutral_factor")]
    coda: Weight,
}

impl From<UntaggedEither<Weight, StressAdjustItemInner>> for StressAdjustItemInner {
    fn from(value: UntaggedEither<Weight, Self>) -> Self {
        match value {
            UntaggedEither::Left(weight) => StressAdjustItemInner {
                onset: weight,
                nucleus: weight,
                coda: weight,
            },
            UntaggedEither::Right(weights) => weights,
        }
    }
}

type StressAdjustItem =
    DeserializeVia<UntaggedEither<Weight, StressAdjustItemInner>, StressAdjustItemInner>;

#[derive(Default, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct StressAdjust {
    #[serde(default)]
    primary: StressAdjustItem,
    #[serde(default)]
    secondary: StressAdjustItem,
    #[serde(default)]
    none: StressAdjustItem,
}

type FullRuleGroupViolations = HashMap<String, HashMap<String, TomlOption<Weight>>>;

#[derive(Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct Weights {
    #[serde(default)]
    output_overlap: TomlOption<Weight>,
    #[serde(default)]
    order_violation: OrderViolationWeights,
    #[serde(default)]
    rule_group_violation: TomlOption<UntaggedEither<Weight, FullRuleGroupViolations>>,
    #[serde(default)]
    skip_group: Weight,
    #[serde(default)]
    double_match: TomlOption<Weight>,
    #[serde(default)]
    add_rule: Weight,
    #[serde(default)]
    alternative: Weight,
    #[serde(default)]
    disambiguator: Weight,
    #[serde(default)]
    anti_disambiguator: Weight,

    #[serde(default)]
    ipa: IpaWeights,
    #[serde(default)]
    ortho: OrthoWeights,
    #[serde(default)]
    syllables: SyllableWeights,
    #[serde(default)]
    stress_adjust: StressAdjust,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct SystemInner {
    language: PathBuf,
    #[serde(default = "return_false")]
    with_phonetic_only: bool,
    layout: Layout,
    rules: Rules,
    weights: Weights,
}

#[derive(Debug)]
pub struct System {
    inner: SystemInner,
    language: Language,
    unordered: Chord,
    disambiguators: Vec<usize>,
    compiled_rules: Vec<Vec<CompiledRule>>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
enum SyllablePosition {
    Onset,
    Nucleus,
    Coda,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct Inspected {
    included_new_match: bool,
    first_inspected: usize,
}

impl Inspected {
    fn at(first_inspected: usize) -> Self {
        Inspected {
            included_new_match: false,
            first_inspected,
        }
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct MatcherData {
    matched: SmallBitSet,
    inspected: Option<Inspected>,
}

type DisallowedRulesList<'a> = RefList<'a, (usize, usize), HashRefList>;

#[derive(Default)]
pub struct Arenas<'a, 'b> {
    disallowed_rules_arena: Arena<(DisallowedRulesList<'b>, (usize, usize))>,
    rule_history_arena: Arena<(RefList<'b, &'a Chord, HashRefList>, &'a Chord)>,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct SplitInfo {
    syllable_range: ops::Range<usize>,
    next_allowed_match: usize,
    matched_from_allowed: SmallBitSet,
}

#[derive(Clone, Debug)]
struct ChordCandidate<'a> {
    weight: Weight,
    chord: Chord,
    previous: Option<SplitInfo>,
    rule_history: Vec<&'a Chord>,
}

type MatcherCacheKey<'a> = (PtrOrd<&'a CompiledRule>, usize, usize, SmallBitSet);

#[derive(Educe)]
#[educe(Debug(bound = "'static: 'a, 'static: 'b"))]
struct WordContext<'a, 'b> {
    this: &'a System,
    word: &'a PreparedWord,
    input: MixedInput<'a>,
    #[educe(Debug(ignore))]
    arenas: &'b Arenas<'a, 'b>,

    syllable_map: Vec<(StressMarker, SyllablePosition)>,
    ipa_costs: matcher::Costs<Weight>,
    ortho_factor: Weight,
    ortho_costs: matcher::Costs<Weight>,

    #[educe(Debug(ignore))]
    chord_alternatives: Cell<HashMap<SplitInfo, Vec<ChordCandidate<'a>>>>,
    #[educe(Debug(ignore))]
    matcher_cache: Cell<HashMap<MatcherCacheKey<'a>, Option<(Weight, MatcherData)>>>,
}

impl<'a, 'b> WordContext<'a, 'b> {
    fn new(this: &'a System, word: &'a PreparedWord, arenas: &'b Arenas<'a, 'b>) -> Self {
        let ortho_factor = this
            .inner
            .weights
            .ortho
            .factor(word.ipa.len(), word.ortho.chars().count());

        WordContext {
            this,
            word,
            input: MixedInput {
                left: &word.ipa,
                right: &word.ortho,
                assoc: &word.ortho_mapping,
            },
            arenas,

            syllable_map: word
                .stress_markers
                .iter()
                .zip(
                    word.stress_markers
                        .iter()
                        .skip(1)
                        .map(|&(idx, _)| idx)
                        .chain(iter::once(word.ipa.len())),
                )
                .flat_map(|(&(from, marker), to)| {
                    let nucleus_start = (from..to)
                        .find(|&idx| word.ipa[idx].syllabic())
                        .unwrap_or(to);
                    (from..to).map(move |idx| {
                        (
                            marker,
                            if idx < nucleus_start {
                                SyllablePosition::Onset
                            } else if word.ipa[idx].syllabic() {
                                SyllablePosition::Nucleus
                            } else {
                                SyllablePosition::Coda
                            },
                        )
                    })
                })
                .collect::<Vec<_>>(),

            ipa_costs: this.inner.weights.ipa.costs(),
            ortho_factor,
            ortho_costs: this.inner.weights.ortho.costs(ortho_factor),

            chord_alternatives: Cell::new(HashMap::default()),
            matcher_cache: Cell::new(HashMap::default()),
        }
    }

    fn stress_adjust(&self, idx: usize) -> Weight {
        let (stress, pos) = self.syllable_map[idx];
        let by_stress = &self.this.inner.weights.stress_adjust;
        let by_pos = match stress {
            StressMarker::None => &by_stress.none,
            StressMarker::Secondary => &by_stress.secondary,
            StressMarker::Primary => &by_stress.primary,
        };
        match pos {
            SyllablePosition::Onset => by_pos.0.onset,
            SyllablePosition::Nucleus => by_pos.0.nucleus,
            SyllablePosition::Coda => by_pos.0.coda,
        }
    }

    fn ipa_len(&self) -> usize {
        self.word.ipa.len()
    }

    fn empty_chord(&self) -> Chord {
        self.this.inner.layout.empty_chord()
    }

    fn syllable_len(&self) -> usize {
        self.word.stress_markers.len()
    }

    fn syllable_pos(&self, syllable: usize) -> usize {
        self.word
            .stress_markers
            .get(syllable)
            .map_or_else(|| self.ipa_len(), |x| x.0)
    }

    #[allow(clippy::option_option)] // makes sense in this case
    fn cached_match_result(
        &self,
        state: &TranslationState<'a, 'b>,
        rule: &'a CompiledRule,
    ) -> Option<Option<(Weight, MatcherData)>> {
        let map = self.matcher_cache.take();
        let out = map
            .get(&(
                PtrOrd(rule),
                state.next_allowed_match,
                state.syllable_range.end,
                state.matched.clone(),
            ))
            .cloned();
        self.matcher_cache.set(map);

        out
    }

    fn cache_match_result(
        &self,
        state: &TranslationState<'a, 'b>,
        rule: &'a CompiledRule,
        out: Option<(Weight, MatcherData)>,
    ) {
        let mut map = self.matcher_cache.take();
        map.insert(
            (
                PtrOrd(rule),
                state.next_allowed_match,
                state.syllable_range.end,
                state.matched.clone(),
            ),
            out,
        );
        self.matcher_cache.set(map);
    }

    fn mixed_index_from_ipa(&self, idx: usize) -> MixedInputState<usize, usize, usize> {
        MixedInputState {
            left: idx,
            right: self.input.left_to_right_wide(&idx).0.unwrap_or(0),
            assoc: 0,
        }
    }
}

#[must_use]
#[derive(PartialEq, Eq)]
enum CommitChordResult {
    Continue,
    Fail,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct TranslationState<'a, 'b> {
    matched: SmallBitSet,
    next_allowed_match: usize,

    previous: Option<SplitInfo>,

    syllable_range: ops::Range<usize>,

    rule_history: RefList<'b, &'a Chord, HashRefList>,
}

impl<'a, 'b> TranslationState<'a, 'b> {
    fn new(ctx: &WordContext<'a, 'b>) -> Self {
        TranslationState {
            matched: SmallBitSet::with_capacity(ctx.ipa_len()),
            next_allowed_match: 0,

            previous: None,

            syllable_range: 0..1,

            rule_history: RefList::Beginning,
        }
    }

    fn can_join_syllables(&self, ctx: &WordContext<'a, 'b>) -> bool {
        self.syllable_range.end < ctx.syllable_len()
    }

    #[allow(clippy::unused_self)]
    fn join_weight(&self, ctx: &WordContext<'a, 'b>) -> Option<Weight> {
        ctx.this.inner.weights.syllables.join.0
    }

    fn join(&mut self, _: &WordContext<'a, 'b>) {
        self.syllable_range.end += 1;
    }

    fn join_drop_weight(
        &self,
        ctx: &WordContext<'a, 'b>,
        drop_first: bool,
        drop_fully: bool,
    ) -> Option<Weight> {
        let syllable = self.syllable_range.end - usize::from(drop_first);

        let stress = ctx.word.stress_markers[syllable].1;

        if drop_fully {
            let weights = &ctx.this.inner.weights.syllables.drop_fully.0;

            match stress {
                StressMarker::None => weights.none.0,
                StressMarker::Secondary => weights.secondary.0,
                StressMarker::Primary => weights.primary.0,
            }
        } else {
            let weights = &ctx.this.inner.weights.syllables.join_drop.0;
            let directional = match stress {
                StressMarker::None => &weights.none.0,
                StressMarker::Secondary => &weights.secondary.0,
                StressMarker::Primary => &weights.primary.0,
            };

            if drop_first {
                directional.left.0
            } else {
                directional.right.0
            }
        }
    }

    fn join_drop(&mut self, ctx: &WordContext<'a, 'b>, drop_first: bool, drop_fully: bool) {
        let syllable = self.syllable_range.end - usize::from(drop_first);

        let range_of_syllable = ctx.syllable_pos(syllable)..ctx.syllable_pos(syllable + 1);
        let iter = ctx
            .word
            .ipa
            .iter()
            .enumerate()
            .take(range_of_syllable.end)
            .skip(range_of_syllable.start);

        for (idx, elem) in iter {
            if elem.syllabic() || !elem.consonant() || drop_fully {
                self.matched.set(idx);
            }
        }

        self.join(ctx);
    }

    fn lost_ipa_cost(&self, ctx: &WordContext<'a, 'b>, until: usize) -> Option<Weight> {
        let lost = (self.next_allowed_match..until)
            .filter(|&i| !self.matched.bit(i))
            .map(|idx| ctx.stress_adjust(idx))
            .sum::<Weight>();

        ctx.ipa_costs
            .ignored
            .map(|ignore_cost| ignore_cost * lost)
            .or_else(|| (lost == 0.0).then_some(lost))
    }

    fn commit_chord_weight(&self, ctx: &WordContext<'a, 'b>) -> Option<Weight> {
        self.lost_ipa_cost(ctx, ctx.syllable_pos(self.syllable_range.end))
    }

    fn commit_chord(
        &mut self,
        ctx: &WordContext<'a, 'b>,
        chord_state: &mut ChordState<'b>,
        weight: Weight,
    ) -> CommitChordResult {
        let chord_state = mem::replace(chord_state, ChordState::new(ctx));

        let info = SplitInfo {
            syllable_range: self.syllable_range.clone(),
            next_allowed_match: self.next_allowed_match,
            matched_from_allowed: self.matched.shifted_dropped_lower(self.next_allowed_match),
        };

        let mut chord_alternatives = ctx.chord_alternatives.take();
        let entry = chord_alternatives.entry(info.clone());

        let was_occupied = matches!(entry, Entry::Occupied(_));

        let mut rule_history = self.rule_history.rev_iter().copied().collect::<Vec<_>>();
        rule_history.reverse();
        entry.or_default().push(ChordCandidate {
            weight,
            chord: chord_state.chord,
            previous: self.previous.clone(),
            rule_history,
        });
        self.previous = Some(info);
        self.rule_history = RefList::Beginning;

        ctx.chord_alternatives.set(chord_alternatives);

        if was_occupied {
            CommitChordResult::Fail
        } else {
            CommitChordResult::Continue
        }
    }

    fn split_between_weight(&self, ctx: &WordContext<'a, 'b>) -> Option<Weight> {
        ctx.this
            .inner
            .weights
            .syllables
            .split_between
            .0
            .and_then(|l| self.commit_chord_weight(ctx).map(|r| l + r))
    }

    #[allow(clippy::range_plus_one)] // doesn't fit the type otherwise
    fn split_between(
        &mut self,
        ctx: &WordContext<'a, 'b>,
        chord_state: &mut ChordState<'b>,
        weight: Weight,
    ) -> CommitChordResult {
        self.syllable_range = self.syllable_range.end..(self.syllable_range.end + 1);
        self.next_allowed_match = ctx.syllable_pos(self.syllable_range.start);
        self.commit_chord(ctx, chord_state, weight)
    }

    #[allow(clippy::unused_self)]
    fn split_inside_weight(&self, ctx: &WordContext<'a, 'b>) -> Option<Weight> {
        ctx.this.inner.weights.syllables.split_inside.0
    }

    fn split_inside(
        &mut self,
        ctx: &WordContext<'a, 'b>,
        chord_state: &mut ChordState<'b>,
        weight: Weight,
    ) -> CommitChordResult {
        let out = self.commit_chord(ctx, chord_state, weight);
        chord_state.current_group = ctx.this.inner.weights.syllables.split_inside_start.as_ref().map(|s| {
            GroupIndex(ctx.this.inner.rules.order.iter().position(|x| x == s).unwrap())
        });
        out
    }

    fn split_nucleus_weight(&self, ctx: &WordContext<'a, 'b>) -> Option<Weight> {
        if self.syllable_range.len() < 2 {
            return None;
        }

        let weights = &ctx.this.inner.weights.syllables.split_at_nucleus.0;

        let mut weight = match ctx.word.stress_markers[self.syllable_range.end - 1].1 {
            StressMarker::None => weights.none.0?,
            StressMarker::Secondary => weights.secondary.0?,
            StressMarker::Primary => weights.primary.0?,
        };

        let mut nucleus_pos = None;
        let syllable_end = ctx.syllable_pos(self.syllable_range.end);
        for i in ctx.syllable_pos(self.syllable_range.end - 1)..syllable_end {
            let nucleus_or_later = ctx.syllable_map[i].1 != SyllablePosition::Onset;
            if nucleus_or_later && self.matched.bit(i) {
                return None;
            }

            if nucleus_pos.is_none() && (nucleus_or_later || !ctx.word.ipa[i].consonant()) {
                nucleus_pos = Some(i);
            }
        }

        let nucleus_pos = nucleus_pos.unwrap_or(syllable_end);
        if nucleus_pos > self.next_allowed_match {
            weight += self.lost_ipa_cost(ctx, nucleus_pos)?;
        }

        Some(weight)
    }

    #[allow(clippy::range_plus_one)] // doesn't fit the type otherwise
    fn split_nucleus(
        &mut self,
        ctx: &WordContext<'a, 'b>,
        chord_state: &mut ChordState<'b>,
        weight: Weight,
    ) -> CommitChordResult {
        let mut coda_pos = None;
        let syllable_end = ctx.syllable_pos(self.syllable_range.end);
        for i in ctx.syllable_pos(self.syllable_range.end - 1)..syllable_end {
            if ctx.syllable_map[i].1 == SyllablePosition::Coda {
                coda_pos = Some(i);
            }
        }

        self.next_allowed_match = coda_pos.unwrap_or(syllable_end);
        self.syllable_range = (self.syllable_range.end - 1)..(self.syllable_range.end + 1);
        self.commit_chord(ctx, chord_state, weight)
    }

    fn matcher_data(&self) -> MatcherData {
        MatcherData {
            matched: self.matched.clone(),
            inspected: None,
        }
    }

    fn advance_ipa<'c>(
        &self,
        ctx: &WordContext<'a, 'b>,
        rule: &'a CompiledRule,
    ) -> impl Fn(
        &WordContext<'a, 'b>,
        &mut MatcherData,
        ItemMatchInfo<&[ipa::Element], Option<ipa::Element>>,
    ) -> Option<Weight>
           + Clone
           + 'c
    where
        'c: 'a,
        'b: 'c,
    {
        let syllable_end = ctx.syllable_pos(self.syllable_range.end);

        move |ctx, data, match_info| match match_info {
            ItemMatchInfo::Ignored { input } => {
                let matched = data.matched.bit(input.before);
                let dropped = matched && (input.item.syllabic() || !input.item.consonant());
                if (data.inspected.is_some() && !dropped) || !matched {
                    Some(ctx.ipa_costs.ignored? * ctx.stress_adjust(input.before))
                } else {
                    Some(Weight::default())
                }
            },
            ItemMatchInfo::Unmatched { .. } => ctx.ipa_costs.unmatched,
            ItemMatchInfo::Matched {
                pattern,
                consume,
                input,
            } => {
                let inspected = data
                    .inspected
                    .get_or_insert_with(|| Inspected::at(input.before));

                let mut diff = Weight::default();

                if consume {
                    if data.matched.bit(input.before) {
                        diff += ctx.this.inner.weights.double_match.0?;
                    } else {
                        inspected.included_new_match = true;
                        data.matched.set(input.before);
                    }

                    if !rule.syllable_lookahead && input.before >= syllable_end {
                        diff += ctx.this.inner.weights.syllables.split_lookahead.0?;
                    }
                }

                diff += ctx.this.inner.weights.ipa.diff(*pattern, input.item)?;

                Some(diff)
            },
        }
    }

    fn advance_mixed<'c>(
        &self,
        ctx: &WordContext<'a, 'b>,
        rule: &'a CompiledRule,
    ) -> impl Fn(
        &WordContext<'a, 'b>,
        &mut MatcherData,
        ItemMatchInfo<MixedInput, Either<Option<ipa::Element>, Option<char>>>,
    ) -> Option<Weight>
           + Clone
           + 'c
    where
        'c: 'a,
        'b: 'c,
    {
        let advance = self.advance_ipa(ctx, rule);

        move |ctx, data, match_info| match match_info.mixed()? {
            Left(match_info) => advance(ctx, data, match_info),
            Right(ItemMatchInfo::Ignored { .. }) => {
                if data.inspected.is_some() {
                    ctx.ortho_costs.ignored
                } else {
                    Some(Weight::default())
                }
            },
            Right(ItemMatchInfo::Unmatched { .. }) => ctx.ortho_costs.unmatched,
            Right(ItemMatchInfo::Matched { pattern, input, .. }) => {
                data.inspected
                    .get_or_insert_with(|| Inspected::at(match_info.input_state_before().left));

                ctx.this
                    .inner
                    .weights
                    .ortho
                    .diff(*pattern, input.item, ctx.ortho_factor)
            },
        }
    }

    fn next_allowed_mixed(
        &self,
        ctx: &WordContext<'a, 'b>,
    ) -> MixedInputState<usize, usize, usize> {
        ctx.mixed_index_from_ipa(self.next_allowed_match)
    }

    fn unless_weight(&self, ctx: &WordContext<'a, 'b>, unless: &'a UnlessInner, first_inspected: usize) -> Option<Weight> {
        let context = (ctx, self, unless);

        let out = explore::controlled_run(
            unless.matching.trivially_explore_on_mixed(
                ctx.input,
                Some(ctx.mixed_index_from_ipa(first_inspected)),
                (),
                matcher::Costs {
                    ignored: Some(Weight::new(1.0).unwrap()),
                    unmatched: Some(Weight::new(1.0).unwrap()),
                },
                matcher::Costs {
                    ignored: ctx.ortho_costs.ignored,
                    unmatched: Some(Weight::new(1.0).unwrap()),
                },
                |&l, &r| context.0.this.inner.weights.ipa.diff(l, r),
                |&l, &r| {
                    context
                        .0
                        .this
                        .inner
                        .weights
                        .ortho
                        .diff(l, r, context.0.ortho_factor)
                },
                |&mut (), _| (),
                |&mut (), _| (),
            ),
            None,
            |w| {
                if w <= unless.threshold {
                    Ok(None)
                } else {
                    Err(())
                }
            },
        )
        .next();

        match out {
            Some(Ok((weight, ((), matcher::AnyTail, matcher::AnyTail, _))))
                if weight <= unless.threshold =>
            {
                unless.importance
            },
            None | Some(_) => Some(Weight::default()),
        }
    }

    fn match_ipa<'c>(
        &self,
        ctx: Rc<WordContext<'a, 'b>>,
        rule: &'a CompiledRule,
    ) -> impl Explorer<Weight, (usize, usize, MatcherData), Return = MatcherData> + 'c
    where
        'c: 'a,
        'b: 'c,
    {
        let advance = self.advance_ipa(&ctx, rule);
        let ipa = &ctx.word.ipa;

        rule.from.explore_on(
            ipa,
            Some(self.next_allowed_match),
            self.matcher_data(),
            (),
            move |&(), data, match_info| {
                advance(&ctx, data, match match_info {
                    ItemMatchInfo::Ignored { input } => ItemMatchInfo::Ignored { input },
                    ItemMatchInfo::Unmatched {
                        pattern: &Left(ref pattern),
                        consume,
                        input_state,
                    } => ItemMatchInfo::Unmatched {
                        pattern,
                        consume,
                        input_state,
                    },
                    ItemMatchInfo::Matched {
                        pattern: &Left(ref pattern),
                        consume,
                        input,
                    } => ItemMatchInfo::Matched {
                        pattern,
                        consume,
                        input,
                    },
                    ItemMatchInfo::Unmatched {
                        pattern: &Right(_), ..
                    }
                    | ItemMatchInfo::Matched {
                        pattern: &Right(_), ..
                    } => return None,
                })
            },
            |&(), data, _| Some((Weight::default(), data)),
        )
    }

    fn match_mixed<'c>(
        &self,
        ctx: Rc<WordContext<'a, 'b>>,
        rule: &'a CompiledRule,
    ) -> impl Explorer<
        Weight,
        (MixedInputState<usize, usize, usize>, usize, MatcherData),
        Return = MatcherData,
    > + 'c
    where
        'c: 'a,
        'b: 'c,
    {
        let advance = self.advance_mixed(&ctx, rule);
        let input = ctx.input;

        rule.from.explore_on(
            input,
            Some(self.next_allowed_mixed(&ctx)),
            self.matcher_data(),
            (),
            move |&(), data, match_info| advance(&ctx, data, match_info),
            |&(), data, _| Some((Weight::default(), data)),
        )
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct GroupIndex(usize);

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct ChordState<'b> {
    chord: Chord,
    blocked: Chord,
    order_violation: usize,
    current_group: Option<GroupIndex>,
    disallowed_rules: RefList<'b, (usize, usize), HashRefList>,
}

impl<'b> ChordState<'b> {
    fn new(ctx: &WordContext<'_, 'b>) -> Self {
        ChordState {
            chord: ctx.empty_chord(),
            blocked: ctx.empty_chord(),
            order_violation: 0,
            current_group: None,
            disallowed_rules: RefList::Beginning,
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn possible_groups<'a>(
        &self,
        ctx: &WordContext<'a, 'b>,
    ) -> impl Iterator<Item = (Weight, GroupIndex)> + Clone + 'a {
        let inner = &ctx.this.inner;
        let current_group = self.current_group.clone().map(|x| x.0);
        (0..ctx.this.compiled_rules.len()).filter_map(move |group| {
            Some(
                if current_group
                    .map_or(0..=0, |g| g..=(g + 1))
                    .contains(&group)
                {
                    (Weight::default(), GroupIndex(group))
                } else if group > current_group.unwrap_or(0) {
                    (
                        inner.weights.skip_group
                            * Weight::new((group - current_group.map_or(0, |g| g + 1)) as f32)
                                .unwrap(),
                        GroupIndex(group),
                    )
                } else {
                    (
                        match *inner.weights.rule_group_violation.0.as_ref()? {
                            UntaggedEither::Left(l) => l,
                            UntaggedEither::Right(ref map) => {
                                map.get(&inner.rules.order[current_group.unwrap_or(0)])?
                                    .get(&inner.rules.order[group])?
                                    .0?
                            },
                        },
                        GroupIndex(group),
                    )
                },
            )
        })
    }

    fn select_group(&mut self, group: GroupIndex) {
        if self.current_group.clone().map_or(true, |x| group.0 > x.0) {
            self.current_group = Some(group);
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    fn possible_rules<'a>(
        &self,
        ctx: &WordContext<'a, 'b>,
        group: GroupIndex,
    ) -> impl Iterator<Item = &'a CompiledRule> + Clone + 'b {
        let this = &ctx.this;
        let disallowed_rules = self.disallowed_rules;

        this.compiled_rules[group.0]
            .iter()
            .enumerate()
            .filter(move |&(idx, _)| !disallowed_rules.rev_iter().any(|&i| (group.0, idx) == i))
            .map(|(_, rule)| rule)
    }

    #[allow(clippy::unused_self)] // since the alternative rule weight is also here
    fn rule_unless_weight<'a>(
        &self,
        ctx: &WordContext<'a, 'b>,
        state: &TranslationState<'a, 'b>,
        rule: &'a CompiledRule,
        first_inspected: usize,
    ) -> Option<Weight> {
        let mut out = Weight::default();

        for i in &rule.unless {
            out += state.unless_weight(ctx, i, first_inspected)?;
        }

        Some(out)
    }

    fn rule_alternative_weight(
        &mut self,
        ctx: &WordContext<'_, 'b>,
        rule: &CompiledRule,
        alternative: &Alternative,
    ) -> Option<Weight> {
        let mut out = Weight::default();

        if alternative.blocked.overlaps_with(&self.blocked) {
            out += ctx.this.inner.weights.output_overlap.0?;
        }

        if !rule.unordered
            && !alternative
                .to
                .starts_after(&self.chord, &ctx.this.unordered)
        {
            out += ctx.this.inner.weights.order_violation.0 .0[self.order_violation].0?;
            if self.order_violation + 1 < ctx.this.inner.weights.order_violation.0 .0.len() {
                self.order_violation += 1;
            }
        }

        Some(out)
    }

    fn chord_update<'a>(
        &mut self,
        ctx: &WordContext<'a, 'b>,
        state: &mut TranslationState<'a, 'b>,
        rule: &'a CompiledRule,
        alternative: &'a Alternative,
        matched: SmallBitSet,
        first_inspected: usize,
    ) {
        state.matched = matched;
        state.next_allowed_match = first_inspected + 1;

        self.chord.extend(&alternative.to);
        self.blocked.extend(&alternative.blocked);

        for &i in &rule.excludes {
            self.disallowed_rules
                .add(i, &ctx.arenas.disallowed_rules_arena);
        }

        if rule.force_next_group {
            self.current_group = Some(GroupIndex(
                self.current_group.clone().map_or(0, |i| i.0 + 1),
            ));
        }

        state
            .rule_history
            .add(&rule.main.to, &ctx.arenas.rule_history_arena);
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct TranslationCandidate {
    pub translation: Vec<ChordString>,
    pub rule_history: Vec<ChordString>,
}

#[derive(Clone)]
struct TranslationCandidateAccumulation {
    previous: Option<SplitInfo>,
    translation: Vec<Chord>,
    rule_history: Vec<ChordString>,
    explored: usize,
}

enum TranslationQueueItem {
    ExploredNext,
    Accumulate(TranslationCandidateAccumulation),
    Final(TranslationCandidate),
}

type ExplorationItem = (Weight, Option<SplitInfo>);

struct PossibleTranslationsIterator<'a, 'b> {
    word_context: Rc<WordContext<'a, 'b>>,
    queue: BinaryHeap<WithOrdOf<Reverse<Weight>, TranslationQueueItem>>,
    // has to be `dyn` because we otherwise hit an ICE as of
    // `rustc 1.76.0-nightly (1e9dda77b 2023-11-22)`:
    exploration: Box<dyn Iterator<Item = ExplorationItem> + 'a>,
    // we might discover more:
    exhausted_accumulations: Vec<(Weight, TranslationCandidateAccumulation)>,
}

impl<'a> PossibleTranslationsIterator<'a, '_> {
    fn explore_next(&mut self) {
        if let Some((next_weight, next)) = self.exploration.next() {
            if let Some(split_info) = next {
                let known_split_info = self
                    .exhausted_accumulations
                    .iter()
                    .any(|i| i.1.previous.as_ref().map_or(false, |i| *i == split_info));

                if !known_split_info {
                    self.queue.push(WithOrdOf(
                        Reverse(next_weight),
                        TranslationQueueItem::Accumulate(TranslationCandidateAccumulation {
                            previous: Some(split_info),
                            translation: Vec::new(),
                            rule_history: Vec::new(),
                            explored: 0,
                        }),
                    ));
                }
            }

            self.queue.push(WithOrdOf(
                Reverse(next_weight),
                TranslationQueueItem::ExploredNext,
            ));

            self.queue.extend(
                self.exhausted_accumulations
                    .drain(..)
                    .map(|(w, i)| WithOrdOf(Reverse(w), TranslationQueueItem::Accumulate(i))),
            );
        }
    }

    fn finalize(&mut self, weight: Weight, accumulation: &TranslationCandidateAccumulation) {
        self.queue.extend(
            self.word_context
                .this
                .disambiguators
                .iter()
                .copied()
                .powerset()
                .map(|v| {
                    let mut chord = accumulation.translation.last().unwrap().clone();
                    let mut weight = weight;
                    for i in v {
                        let weights = &self.word_context.this.inner.weights;
                        if !chord.0.bit(i) {
                            chord.0.set(i);
                            weight += weights.disambiguator;
                        } else {
                            chord.0.unset(i);
                            weight += weights.anti_disambiguator;
                        }
                    }

                    WithOrdOf(
                        Reverse(weight),
                        TranslationQueueItem::Final(TranslationCandidate {
                            translation: accumulation.translation
                                [..accumulation.translation.len() - 1]
                                .iter()
                                .chain(iter::once(&chord))
                                .map(|chord| self.word_context.this.inner.layout.render(chord))
                                .collect(),
                            rule_history: accumulation.rule_history.clone(),
                        }),
                    )
                }),
        );
    }

    fn accumulate(
        &mut self,
        weight: Weight,
        accumulation: TranslationCandidateAccumulation,
        choices: &[ChordCandidate<'a>],
    ) {
        if accumulation.explored == choices.len() {
            self.exhausted_accumulations.push((weight, accumulation));
        } else {
            let choice = &choices[accumulation.explored];
            let next_weight =
                weight + choice.weight - choices[accumulation.explored.saturating_sub(1)].weight;

            self.queue.extend([
                WithOrdOf(
                    Reverse(next_weight),
                    TranslationQueueItem::Accumulate(TranslationCandidateAccumulation {
                        previous: choice.previous.clone(),
                        translation: iter::once(&choice.chord)
                            .chain(&accumulation.translation)
                            .cloned()
                            .collect(),
                        rule_history: choice
                            .rule_history
                            .iter()
                            .map(|&chord| self.word_context.this.inner.layout.render(chord))
                            .chain(accumulation.rule_history.iter().cloned())
                            .collect(),
                        explored: 0,
                    }),
                ),
                WithOrdOf(
                    Reverse(next_weight),
                    TranslationQueueItem::Accumulate(TranslationCandidateAccumulation {
                        explored: accumulation.explored + 1,
                        ..accumulation
                    }),
                ),
            ]);
        }
    }
}

impl Iterator for PossibleTranslationsIterator<'_, '_> {
    type Item = (Weight, TranslationCandidate);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let WithOrdOf(Reverse(weight), item) = self.queue.pop()?;
            match item {
                TranslationQueueItem::ExploredNext => {
                    self.explore_next();
                    continue;
                },
                TranslationQueueItem::Final(out) => break Some((weight, out)),
                TranslationQueueItem::Accumulate(accumulation) => {
                    match accumulation.previous {
                        None => self.finalize(weight, &accumulation),
                        Some(ref split_info) => {
                            let map = self.word_context.chord_alternatives.take();
                            let choices = &map[split_info];
                            self.accumulate(weight, accumulation, choices);
                            self.word_context.chord_alternatives.set(map);
                        },
                    }
                    continue;
                },
            }
        }
    }
}

impl System {
    fn load_toml<T: DeserializeOwned>(path: impl AsRef<Path>) -> T {
        toml::from_str(&fs::read_to_string(path).unwrap()).unwrap()
    }

    #[must_use]
    pub fn load(path: impl AsRef<Path>) -> Self {
        let inner = Self::load_toml::<SystemInner>(&path);

        let language_path = path
            .as_ref()
            .parent()
            .unwrap_or_else(|| ".".as_ref())
            .join(&inner.language);

        let disambiguators_chord = inner.layout.parse(&inner.layout.disambiguators);

        let unordered = inner.layout.parse(&inner.layout.unordered);

        System {
            disambiguators: (0..disambiguators_chord.0.capacity())
                .filter(|&i| disambiguators_chord.0.bit(i))
                .collect(),
            compiled_rules: inner.rules.compile(&inner.layout, &unordered),
            unordered,
            inner,
            language: Self::load_toml(language_path),
        }
    }

    #[must_use]
    pub fn language(&self) -> &Language {
        &self.language
    }

    fn prepare_middle(
        &self,
        ipa: Vec<ipa::Element>,
        ortho: &str,
        stress_markers: Vec<(usize, StressMarker)>,
    ) -> PreparedWord {
        let mut out = PreparedWord {
            ortho_mapping: self.language.map_ortho(&ipa, ortho),
            ipa,
            ortho: ortho.to_owned(),
            stress_markers,
        };

        self.language
            .split_syllables(&mut out.stress_markers, MixedInput {
                left: &out.ipa,
                right: &out.ortho,
                assoc: &out.ortho_mapping,
            });

        out
    }

    #[must_use]
    pub fn prepare_ipa(&self, ipa: &str, ortho: &str) -> PreparedWord {
        let mut ipa = self.language.ipa_with_substitutions(ipa.to_owned());
        let stress_markers = ipa::extract_syllables(&mut ipa);
        self.prepare_middle(ipa, ortho, stress_markers)
    }

    #[must_use]
    pub fn prepare_arpa(&self, arpa: &str, ortho: &str) -> PreparedWord {
        let ipa = arpabet::to_ipa(arpa);

        let mut out = self.prepare_middle(ipa.ipa, ortho, ipa.stress_markers);

        for &mut (idx, ref mut kind) in &mut out.stress_markers {
            *kind = ipa
                .vowel_stress
                .iter()
                .find(|&&(vowel_idx, _)| vowel_idx >= idx)
                .unwrap_or(&(0, StressMarker::None))
                .1;
        }

        out
    }

    #[allow(clippy::too_many_lines)]
    pub fn possible_translations<'a>(
        &'a self,
        word: &'a PreparedWord,
        max_size: Option<usize>,
        arenas: &'a Arenas<'a, 'a>,
    ) -> impl Iterator<Item = (Weight, TranslationCandidate)> + 'a {
        let ctx = Rc::new(WordContext::new(self, word, arenas));
        let ctx_clone = Rc::clone(&ctx);

        #[allow(clippy::cognitive_complexity)]
        let explorer = #[coroutine] move |_| {
            let mut translation_state = TranslationState::new(&ctx);
            let mut chord_state = ChordState::new(&ctx);

            let split_info = loop {
                if chord_state.chord.is_empty() {
                    if translation_state.can_join_syllables(&ctx) && explore::branch!(binary) {
                        if explore::branch!(binary) {
                            // Join
                            explore::try_push!(translation_state.join_weight(&ctx));
                            translation_state.join(&ctx);
                        } else {
                            let drop_first = explore::branch!(binary);
                            let drop_fully = explore::branch!(binary);
                            explore::try_push!(
                                translation_state.join_drop_weight(&ctx, drop_first, drop_fully),
                            );
                            translation_state.join_drop(&ctx, drop_first, drop_fully);
                        }
                        continue;
                    }
                } else if explore::branch!(binary) {
                    // potentially be done with this chord

                    macro_rules! commit {
                        ($weight:ident, $commit:ident $(,)?) => {{
                            let weight = NoCopy(
                                (yield match translation_state.$weight(&ctx) {
                                    None => explore::Instruction::Fail,
                                    Some(w) => explore::Instruction::Push(w),
                                })
                                .weight,
                            );

                            let result =
                                translation_state.$commit(&ctx, &mut chord_state, weight.consume());

                            if result == CommitChordResult::Fail {
                                yield explore::Instruction::Fail;
                            }
                        }};
                    }

                    if explore::branch!(binary) {
                        if translation_state.can_join_syllables(&ctx) {
                            if explore::branch!(binary) {
                                commit!(split_between_weight, split_between);
                            } else {
                                commit!(split_nucleus_weight, split_nucleus);
                            }
                            continue;
                        } else {
                            commit!(commit_chord_weight, commit_chord);
                            break translation_state.previous.clone().unwrap();
                        }
                    } else {
                        commit!(split_inside_weight, split_inside);
                        continue;
                    }
                }

                yield explore::Instruction::Dedup((translation_state.clone(), chord_state.clone()));
                yield explore::Instruction::Push(ctx.this.inner.weights.add_rule);

                let group = explore::branch!(weighted: chord_state.possible_groups(&ctx));
                chord_state.select_group(group.clone());

                let rule = explore::branch!(free: chord_state.possible_rules(&ctx, group));

                yield explore::Instruction::Push(rule.add_weight);

                let MatcherData { matched, inspected } = {
                    let out = ctx
                        .cached_match_result(&translation_state, rule)
                        .unwrap_or_else(|| {
                            let mixed = explore::run(
                                translation_state.match_mixed(Rc::clone(&ctx), rule),
                                None,
                            )
                            .next();

                            let out = if ctx.this.inner.with_phonetic_only {
                                let phonetic = explore::run(
                                    translation_state.match_ipa(Rc::clone(&ctx), rule),
                                    None,
                                )
                                .next();

                                match (mixed, phonetic) {
                                    (None, None) => None,
                                    (Some(mixed), None) => Some(mixed),
                                    (Some(mixed), Some(phonetic)) if mixed.0 < phonetic.0 => {
                                        Some(mixed)
                                    },
                                    (None | Some(_), Some(phonetic)) => Some(phonetic),
                                }
                            } else {
                                mixed
                            };

                            ctx.cache_match_result(&translation_state, rule, out.clone());

                            out
                        });

                    match out.map(|(w, o)| (NoCopy(w), o)) {
                        None => explore::fail!(),
                        Some((weight, out)) => {
                            yield explore::Instruction::Push(weight.consume());
                            out
                        },
                    }
                };

                let Some(Inspected {
                    included_new_match: true,
                    first_inspected,
                }) = inspected
                else {
                    explore::fail!()
                };

                explore::try_push!(
                    chord_state.rule_unless_weight(&ctx, &translation_state, rule, first_inspected)
                );

                let alternative = if explore::branch!(weighted: [
                    (Weight::default(), true),
                    (ctx.this.inner.weights.alternative, false)
                ]) {
                    &rule.main
                } else {
                    explore::branch!(free: &rule.alternatives)
                };

                explore::try_push!(chord_state.rule_alternative_weight(&ctx, rule, alternative));

                chord_state.chord_update(
                    &ctx,
                    &mut translation_state,
                    rule,
                    alternative,
                    matched,
                    first_inspected,
                );
            };

            split_info
        };

        // The correction factor (3.4) was calculated experimentally using the word
        // "supercalifragilisticexpialidocious". The problem is that we're holding onto
        // a lot of state *outside* of the actual explorer itself via `Rc`s, but
        // we want the actual limit to be at least in the same ballpark as the
        // one provided by the user
        let state_size = 34 * size_of_val(&explorer) / 10;
        let max_states = max_size.map(|x| x / state_size);

        #[cfg(debug_assertions)]
        #[allow(clippy::cast_precision_loss)]
        {
            println!("Estimated state size: {:.1}K", state_size as f32 / 1024.);
            max_states.map_or_else(
                || println!("No state count limit"),
                |limit| println!("State count limit: {limit}"),
            );
        }

        /*
        let out = dbg!(explore::run(explorer).next());
        let alts = ctx_clone.chord_alternatives.take();

        let mut previous = Some(&out.as_ref().unwrap().1 .0);
        while let Some(ref state) = previous {
            let first = &alts[&state][0];
            println!("{:?}", self.inner.layout.render(&first.chord));
            previous = first.previous.as_ref();
        }
        */

        PossibleTranslationsIterator {
            word_context: ctx_clone,
            queue: iter::once(WithOrdOf(
                Reverse(Weight::default()),
                TranslationQueueItem::ExploredNext,
            ))
            .collect(),
            exploration: Box::new(explore::staggered_run(
                Weight::new(0.1).unwrap(),
                explorer,
                max_states,
            )),
            exhausted_accumulations: Vec::new(),
        }
        .unique_by(|i| {
            i.1.translation
                .iter()
                .map(|i| i.string.clone())
                .collect::<Vec<_>>()
        })
    }
}
