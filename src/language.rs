use std::{cmp, iter, ptr};

use educe::Educe;
use either::{Left, Right};
use rustc_hash::FxHashMap as HashMap;
use serde::Deserialize;
use typed_arena::Arena;

use crate::{
    call, explore,
    helpers::{let_group, DeserializeVia, RefList},
    ipa,
    matcher::{self, IpaMatcher, MixedMatcher, OrthoMatcher, ReturnTailInputState},
    IpaWeights, OrthoWeights, Weight,
};

#[derive(Deserialize, Debug)]
#[serde(from = "HashMap<String, String>")]
struct Substitutions(Vec<(String, String)>);

impl From<HashMap<String, String>> for Substitutions {
    fn from(value: HashMap<String, String>) -> Self {
        let mut out = value.into_iter().collect::<Vec<_>>();
        out.sort_unstable_by_key(|&(ref from, _)| cmp::Reverse(from.len()));
        Substitutions(out)
    }
}

#[derive(Deserialize, Debug)]
struct SyllableStructure {
    onsets: Vec<MixedMatcher>,
}

impl TryFrom<SyllableStructure> for MixedMatcher {
    type Error = &'static str;

    fn try_from(value: SyllableStructure) -> Result<Self, &'static str> {
        value
            .onsets
            .into_iter()
            .reduce(MixedMatcher::or)
            .ok_or("onsets has to have at least one element")
    }
}

fn ortho_add_match_default() -> Weight {
    Weight::new(0.02).unwrap()
}

#[derive(Clone, Copy, Deserialize, Debug, Educe)]
#[educe(Default)]
#[serde(rename_all = "kebab-case")]
struct MapOrthoWeights {
    #[educe(Default(expression = "ortho_add_match_default()"))]
    #[serde(default = "ortho_add_match_default")]
    add_match: Weight,
}

fn split_syllable_coda_element_default() -> Weight {
    Weight::new(0.1).unwrap()
}

fn split_syllable_coda_element_ortho_default() -> Weight {
    Weight::new(0.01).unwrap() * split_syllable_coda_element_default()
}

#[derive(Clone, Copy, Deserialize, Debug, Educe)]
#[educe(Default)]
#[serde(rename_all = "kebab-case")]
struct SplitSyllableWeights {
    #[educe(Default(expression = "split_syllable_coda_element_default()"))]
    #[serde(default = "split_syllable_coda_element_default")]
    coda_element: Weight,
    #[educe(Default(expression = "split_syllable_coda_element_ortho_default()"))]
    #[serde(default = "split_syllable_coda_element_ortho_default")]
    coda_element_ortho: Weight,
}

#[derive(Clone, Copy, Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct Weights {
    #[serde(default)]
    ipa: IpaWeights,
    #[serde(default)]
    ortho: OrthoWeights,
    #[serde(default)]
    map_ortho: MapOrthoWeights,
    #[serde(default)]
    split_syllables: SplitSyllableWeights,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "kebab-case")]
struct LanguageInner {
    substitutions: Substitutions,
    syllable_structure: DeserializeVia<SyllableStructure, MixedMatcher>,
    orthography: HashMap<OrthoMatcher, IpaMatcher>,
    weights: Weights,
}

type MixedInputRef<'a, 'b, 'c> =
    matcher::MixedInput<&'a [ipa::Element], &'b str, &'c [(usize, usize)]>;

#[derive(Deserialize, Debug)]
#[serde(from = "LanguageInner")]
pub struct Language {
    inner: LanguageInner,
    ortho_match: usize,
    ipa_match: usize,
}

impl From<LanguageInner> for Language {
    fn from(inner: LanguageInner) -> Self {
        Language {
            ortho_match: inner
                .orthography
                .keys()
                .map(|key| key.stats().max_match)
                .max()
                .unwrap_or(0),
            ipa_match: inner
                .orthography
                .values()
                .map(|value| value.stats().max_match)
                .max()
                .unwrap_or(0),
            inner,
        }
    }
}

impl Language {
    fn apply_substitutions(&self, mut ipa: String) -> String {
        loop {
            let mut changed = false;
            for &(ref from, ref to) in &self.inner.substitutions.0 {
                let mut from_idx = 0;
                while let Some(idx) = ipa[from_idx..].find(from) {
                    from_idx += idx;
                    ipa.replace_range(from_idx..(from_idx + from.len()), to);
                    // Don't match the same thing again
                    from_idx += to.chars().next().map_or(0, char::len_utf8);
                    changed = true;
                }
            }

            if !changed {
                break ipa;
            }
        }
    }

    #[must_use]
    pub fn ipa_with_substitutions(&self, ipa: String) -> Vec<ipa::Element> {
        ipa::segments(&self.apply_substitutions(ipa))
            .map(|(_, elem)| elem)
            .collect()
    }

    #[must_use]
    pub fn ipa_mapping(&self, ipa: String) -> (Vec<String>, Vec<ipa::Element>) {
        let ipa = self.apply_substitutions(ipa);
        let mut strs = Vec::with_capacity(ipa.len());
        let mut elems = Vec::with_capacity(ipa.len());
        for (s, elem) in ipa::segments(&ipa) {
            strs.push(s.to_owned());
            elems.push(elem);
        }
        (strs, elems)
    }

    #[must_use]
    pub fn map_ortho<'a>(&'a self, ipa: &'a [ipa::Element], ortho: &'a str) -> Vec<(usize, usize)> {
        let_group! {
            let group;

            let ortho_len = ortho.chars().count();
            let list_arena = Arena::new();

            let match_single = |ipa_idx, ortho_idx| {
                #[coroutine] move |_| {
                    if self.inner.weights.map_ortho.add_match > Weight::default() {
                        yield explore::Instruction::Push(self.inner.weights.map_ortho.add_match);
                    }

                    let (ortho_matcher, ipa_matcher) = explore::branch!(free: &self.inner.orthography);

                    let costs = matcher::Costs {
                        ignored: Some(Weight::new(1.0).unwrap()),
                        unmatched: Some(Weight::new(1.0).unwrap()),
                    };

                    // Matching ortho first because it's MUCH cheaper
                    let ((), ReturnTailInputState(ortho_idx)) = call!(
                        ortho_matcher.trivially_explore_on(
                            ortho,
                            Some(ortho_idx),
                            (),
                            costs,
                            |&l, &r| OrthoWeights::diff1(l, r),
                            |&mut (), _| (),
                            |&mut (), _| (),
                        ),
                        dedup: |x| Left((ptr::addr_of!(*ortho_matcher), x))
                    );

                    let ((), ReturnTailInputState(ipa_idx)) = call!(
                        ipa_matcher.trivially_explore_on(
                            ipa,
                            Some(ipa_idx),
                            (),
                            costs,
                            |&l, &r| self.inner.weights.ipa.diff(l, r),
                            |&mut (), _| (),
                            |&mut (), _| (),
                        ),
                        dedup: |x| Right((ptr::addr_of!(*ipa_matcher), x))
                    );

                    (ipa_idx, ortho_idx)
                }
            };
        }

        let explorer = #[coroutine] |_| {
            let _ = &group;

            let (mut ipa_idx, mut ortho_idx) = (0, 0);
            let mut ortho_count = 0;

            let mut out = RefList::new_basic();

            while ipa_idx < ipa.len() || ortho_idx < ortho.len() {
                yield explore::Instruction::Dedup((ipa_idx, ortho_idx, None));

                #[allow(clippy::cast_precision_loss)]
                yield explore::Instruction::Heuristic({
                    let heuristic_part =
                        |len: usize, idx: usize, max_match| (len - idx).div_ceil(max_match);

                    Weight::new(cmp::max(
                        heuristic_part(ipa.len(), ipa_idx, self.ipa_match),
                        heuristic_part(group.ortho_len, ortho_count, self.ortho_match),
                    ) as f32)
                    .unwrap()
                        * self.inner.weights.map_ortho.add_match
                });

                let (next_ipa, next_ortho) = call!(
                    (group.match_single)(ipa_idx, ortho_idx),
                    dedup: |x| (ipa_idx, ortho_idx, Some(x))
                );

                out.add((next_ipa, next_ortho), &group.list_arena);

                ortho_count += ortho[ortho_idx..next_ortho].chars().count();
                (ipa_idx, ortho_idx) = (next_ipa, next_ortho);
            }

            out
        };

        let list = explore::run(explorer, None).next().unwrap().1;

        list.collect()
    }

    #[allow(
        clippy::cast_precision_loss,
        clippy::suboptimal_flops,
        clippy::too_many_lines
    )]
    pub fn split_syllables(
        &self,
        markers: &mut Vec<(usize, ipa::StressMarker)>,
        input: MixedInputRef,
    ) {
        let mut starts = Vec::new();
        if markers.first().map_or(true, |&(at, _)| at != 0) {
            markers.insert(0, (0, ipa::StressMarker::None));
        }
        let mut marker_idx = 1;
        let mut offset = 0;
        loop {
            let next = (|| {
                let (edge_begin, _) = input
                    .left
                    .iter()
                    .enumerate()
                    .skip(offset)
                    .skip_while(|&(_, i)| !i.syllabic())
                    .skip_while(|&(_, i)| !i.syllabic())
                    .find(|&(_, i)| !i.syllabic())?;

                let (edge_end, _) = input
                    .left
                    .iter()
                    .enumerate()
                    .skip(edge_begin)
                    .find(|&(_, i)| i.syllabic())?;

                if let Some(&(idx, _)) =
                    markers.get(marker_idx).filter(|&&(idx, _)| idx <= edge_end)
                {
                    offset = idx;
                    marker_idx += 1;
                    return Some(None);
                }

                let ortho_end = input.left_to_right_narrow(&edge_end).0.unwrap_or(0);

                // TODO: Since I've implemented "start from state" stuff in matchers, all we'd
                // need to make this unnecessary is changing that to `impl
                // RangeBounds<I::State>` or something self written that also
                // admits `StateCheck`s.
                //
                // Not super important though as this function is very quick because it only
                // goes through very few symbols each time anyway
                starts.clear();
                starts.extend((edge_begin..=edge_end).map(|idx| {
                    let ortho_start = input.left_to_right_wide(&idx).0.unwrap_or(0);

                    #[allow(clippy::suspicious_operation_groupings)]
                    (ortho_start, matcher::MixedInput {
                        left: &input.left[idx..edge_end],
                        right: &input.right[ortho_start..ortho_end],
                        assoc: input
                            .assoc
                            .iter()
                            .copied()
                            .skip_while(|&(l, _)| l <= idx)
                            .take_while(|&(l, r)| {
                                l <= edge_end && r <= ortho_end && (l, r) != (edge_end, ortho_end)
                            })
                            .chain(iter::once((edge_end, ortho_end)))
                            .map(|(l, r)| (l - idx, r - ortho_start))
                            .collect::<Vec<_>>(),
                    })
                }));

                let_group! {
                    let group;

                    let inner = &self.inner;
                    let starts = &starts;
                    let first_ortho_start = starts[0].0;
                    let ortho_factor = self
                        .inner
                        .weights
                        .ortho
                        .factor(edge_end - edge_begin, ortho_end - first_ortho_start);
                }

                let explorer = #[coroutine] |_| {
                    let _ = &group;

                    let (idx, input) = explore::branch!(weighted: group.starts.iter().enumerate().map(
                        |(idx, &(ortho_idx, ref input))| {
                            let ipa_weight = Weight::new(idx as f32).unwrap()
                                * group.inner.weights.split_syllables.coda_element;
                            let ortho_weight = Weight::new((ortho_idx - group.first_ortho_start) as f32)
                                .unwrap()
                                * group.ortho_factor
                                * group.inner.weights.split_syllables.coda_element_ortho;

                            (ipa_weight + ortho_weight, (idx, input))
                        }
                    ));

                    ((), matcher::IgnoreCostTail, matcher::IgnoreCostTail, _) = call!(
                        group.inner.syllable_structure.0.trivially_explore_on_mixed(
                            MixedInputRef {
                                left: input.left,
                                right: input.right,
                                assoc: &*input.assoc
                            },
                            None,
                            (),
                            group.inner.weights.ipa.costs(),
                            group.inner.weights.ortho.costs(group.ortho_factor),
                            |&l, &r| group.inner.weights.ipa.diff(l, r),
                            |&l, &r| group.inner.weights.ortho.diff(l, r, group.ortho_factor),
                            |&mut (), _| (),
                            |&mut (), _| (),
                        ),
                        dedup: |x| (idx, x)
                    );

                    idx
                };

                let out = explore::run(explorer, None).next();
                out.map(|(_, out)| Some(edge_begin + out))
            })();

            match next {
                None => break,
                Some(None) => continue,
                Some(Some(next)) => {
                    offset = next;
                    markers.insert(marker_idx, (next, ipa::StressMarker::None));
                    marker_idx += 1;
                },
            }
        }
    }
}
