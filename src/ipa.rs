use std::{cmp, fmt, iter};

use educe::Educe;
use either::{Either, Left, Right};
use rustc_hash::FxHashMap as HashMap;
use serde::Deserialize;

const BASES_CSV: &str = include_str!("./data/ipa_bases.csv");
const DIACRITICS_YML: &str = include_str!("./data/diacritic_definitions.yml");
const FEATURE_WEIGHTS_CSV: &str = include_str!("./data/feature_weights.csv");

const FEATURE_COUNT: usize = {
    let weights = BASES_CSV.as_bytes();

    let mut columns = 1;
    let mut idx = 0;
    while idx < weights.len() {
        if weights[idx] == b'\n' {
            break;
        } else if weights[idx] == b',' {
            columns += 1;
        }

        idx += 1;
    }

    columns - 1
};

#[derive(Clone, Copy, PartialEq, Eq, Deserialize, Debug)]
enum Feature {
    #[serde(rename = "-")]
    Negative,
    #[serde(rename = "0")]
    Neutral,
    #[serde(rename = "+")]
    Positive,
}

const FEATURES_BITS: usize = 2;
const FEATURE_MASK: u8 = if FEATURES_BITS == 8 {
    0u8.wrapping_sub(1)
} else {
    (1 << FEATURES_BITS) - 1
};

const fn encode_feature(feature: Feature) -> u8 {
    match feature {
        Feature::Negative => 0b00,
        Feature::Neutral => 0b01,
        Feature::Positive => 0b10,
    }
}

const fn decode_feature(value: u8) -> Feature {
    match value {
        0b00 => Feature::Negative,
        0b01 => Feature::Neutral,
        0b10 => Feature::Positive,
        _ => unreachable!(),
    }
}

fn feature_difference(left: Feature, right: Feature) -> u8 {
    encode_feature(left).abs_diff(encode_feature(right))
}

const FEATURES_PER_BYTE: usize = 8usize / FEATURES_BITS;

const FEATURES_BYTE_LENGTH: usize = FEATURE_COUNT.div_ceil(FEATURES_PER_BYTE);

const FEATURE_NEUTRAL_BYTE: u8 = {
    let mut out = 0;
    let mut idx = 0;
    while idx < FEATURES_PER_BYTE {
        out |= encode_feature(Feature::Neutral) << (idx * FEATURES_BITS);
        idx += 1;
    }

    out
};

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Features([u8; FEATURES_BYTE_LENGTH]);

impl Features {
    const NEUTRAL: Self = Features([FEATURE_NEUTRAL_BYTE; FEATURES_BYTE_LENGTH]);

    fn from_features(features: impl IntoIterator<Item = Feature>) -> Self {
        let mut iter = features.into_iter();
        let mut idx = 0;
        let out = [(); FEATURES_BYTE_LENGTH].map(|()| {
            let mut out = 0;
            for i in 0..FEATURES_PER_BYTE {
                if idx < FEATURE_COUNT {
                    out |= encode_feature(iter.next().unwrap()) << (i * FEATURES_BITS);
                }
                idx += 1;
            }
            out
        });
        assert!(iter.next().is_none());
        Features(out)
    }

    fn features(self) -> impl Iterator<Item = Feature> {
        self.0.into_iter().flat_map(|byte| {
            (0..FEATURES_PER_BYTE)
                .map(move |i| decode_feature((byte >> (i * FEATURES_BITS)) & FEATURE_MASK))
        })
    }

    pub fn diff_vector(self, right: Self) -> impl Iterator<Item = u8> {
        self.features()
            .zip(right.features())
            .map(|(l, r)| feature_difference(l, r))
    }

    #[must_use]
    pub fn unweighted_diff(self, right: Self) -> u8 {
        #[allow(clippy::assertions_on_constants)] // meant to be a compile time assertion
        const _: () = assert!(
            FEATURE_COUNT < 128,
            "feature count is more than 127, so the difference won't fit in a byte"
        );

        self.diff_vector(right).sum()
    }

    #[must_use]
    pub fn weighted_diff(self, right: Self) -> f32 {
        self.diff_vector(right)
            .zip(DATA.weights.iter())
            .map(|(diff, &weight)| f32::from(diff) * weight / 2.0)
            .sum()
    }

    #[must_use]
    pub fn syllabic_aware_diff(self, right: Self) -> Option<f32> {
        (self.syllabic() == right.syllabic()).then(|| self.weighted_diff(right))
    }

    #[must_use]
    pub fn overridden_with(self, overriding_with: Features) -> Self {
        Self::from_features(
            self.features()
                .zip(overriding_with.features())
                .map(|(l, r)| match r {
                    Feature::Neutral => l,
                    r => r,
                }),
        )
    }

    pub fn override_with(&mut self, overriding_with: Features) {
        *self = self.overridden_with(overriding_with);
    }

    #[must_use]
    pub fn syllabic(self) -> bool {
        self.features().nth(DATA.syllabic).unwrap() == Feature::Positive
    }

    #[must_use]
    pub fn consonant(self) -> bool {
        self.features().nth(DATA.consonant).unwrap() == Feature::Positive
    }

    fn symbol(self, data: &'static Data) -> Option<(char, &'static str)> {
        data.bases
            .iter()
            .flat_map(|(&c, v)| v.iter().map(move |&(p, f)| (c, p, f)))
            .find(|&(_, _, f)| f == self)
            .map(|(c, p, _)| (c, p))
    }
}

impl fmt::Debug for Features {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Features(")?;
        let data = &*DATA;
        match self.symbol(data) {
            None => {
                let mut first = true;
                for (feature, name) in self.features().zip(data.feature_names.iter()) {
                    if feature != Feature::Neutral {
                        if first {
                            first = false;
                        } else {
                            write!(f, ", ")?;
                        }

                        if feature == Feature::Positive {
                            write!(f, "+")?;
                        } else {
                            write!(f, "-")?;
                        }

                        write!(f, "{name}")?;
                    }
                }
            },
            Some((c, p)) => write!(f, "{c}{p}")?,
        }
        write!(f, ")")?;
        Ok(())
    }
}

impl fmt::Display for Features {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.symbol(&DATA) {
            None => write!(f, "⌧"),
            Some((c, p)) => write!(f, "{c}{p}"),
        }
    }
}

impl Default for Features {
    fn default() -> Self {
        Self::NEUTRAL
    }
}

type FeatureMap = HashMap<char, Vec<(&'static str, Features)>>;

#[derive(Debug)]
struct Data {
    prefixes: FeatureMap,
    bases: FeatureMap,
    suffixes: FeatureMap,
    weights: [f32; FEATURE_COUNT],
    feature_names: Vec<String>,
    syllabic: usize,
    consonant: usize,
}

lazy_static::lazy_static! {
    static ref DATA: Data = {
        #[derive(Deserialize, Debug)]
        #[serde(rename_all = "kebab-case")]
        enum Position {
            Pre,
            Post,
        }

        #[derive(Deserialize, Debug)]
        struct Diacritic<'a> {
            marker: &'a str,
            position: Position,
            content: HashMap<&'a str, Feature>,
        }

        #[derive(Deserialize, Debug)]
        struct Diacritics<'a> {
            #[serde(borrow)]
            diacritics: Vec<Diacritic<'a>>,
        }

        let push_feature_map = |map: &mut HashMap<_, Vec<_>>, ipa: &'static str, features| {
            let first = ipa.chars().next().unwrap();
            let rest = &ipa[first.len_utf8()..];

            map.entry(first)
                .or_default()
                .push((rest, Features::from_features(features)));
        };

        let mut bases_csv = csv::Reader::from_reader(BASES_CSV.as_bytes());

        let header = bases_csv.headers()
            .unwrap()
            .iter()
            .skip(1)
            .map(String::from)
            .collect::<Vec<_>>();
        let header_indices = header.iter()
            .enumerate()
            .map(|(idx, i)| (&**i, idx))
            .collect::<HashMap<_, _>>();

        let mut bases = HashMap::default();
        for i in bases_csv.deserialize::<(String, [Feature;FEATURE_COUNT])>() {
            let (ipa, features) = i.unwrap();
            let ipa = ipa.leak();
            push_feature_map(&mut bases, ipa, features);
            if ipa.contains('\u{0361}') {
                push_feature_map(&mut bases, ipa.replace('\u{0361}', "\u{0320}").leak(), features);
            }
            if ipa.contains('\u{0320}') {
                push_feature_map(&mut bases, ipa.replace('\u{0320}', "\u{0361}").leak(), features);
            }
        }

        for i in bases.values_mut() {
            i.sort_unstable_by_key(|&(prefix, _)| cmp::Reverse(prefix.len()));
        }

        let diacritics = serde_yaml::from_str::<Diacritics>(DIACRITICS_YML).unwrap();

        let mut prefixes = HashMap::default();
        let mut suffixes = HashMap::default();
        for i in diacritics.diacritics {
            let mut features = [Feature::Neutral;FEATURE_COUNT];
            for (k, v) in i.content {
                features[header_indices[k]] = v;
            }

            let target = match i.position {
                Position::Pre => &mut prefixes,
                Position::Post => &mut suffixes,
            };

            push_feature_map(target, i.marker, features);
        }

        let mut weights = csv::Reader::from_reader(FEATURE_WEIGHTS_CSV.as_bytes());

        let weights_header = weights.headers()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(idx, i)| (i.to_owned(), idx))
            .collect::<HashMap<_, _>>();

        let disordered_weights = weights.deserialize::<Vec<f32>>()
            .next()
            .unwrap()
            .unwrap();

        let mut idx = 0;
        let weights = [();FEATURE_COUNT].map(|()| {
            let column = weights_header.get(&header[idx]);
            let out = column.map_or(1.0, |idx| disordered_weights[*idx]);
            idx += 1;
            out
        });

        let weight_sum = weights.into_iter().sum::<f32>();

        let weights = weights.map(|weight| weight / weight_sum);

        Data {
            prefixes,
            bases,
            suffixes,
            weights,
            syllabic: header_indices["syl"],
            consonant: header_indices["cons"],
            feature_names: header,
        }
    };
}

pub fn load_data_early() {
    let _ = &*DATA;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Educe)]
#[educe(Default)]
pub enum Element {
    #[educe(Default)]
    Features(Features),
    Other(char),
}

impl Element {
    #[must_use]
    pub fn to_either(self) -> Either<Features, char> {
        match self {
            Element::Features(features) => Left(features),
            Element::Other(c) => Right(c),
        }
    }

    #[must_use]
    pub fn features(self) -> Option<Features> {
        self.to_either().left()
    }

    #[must_use]
    pub fn other(self) -> Option<char> {
        self.to_either().right()
    }

    #[must_use]
    pub fn syllabic(self) -> bool {
        self.features().map_or(false, Features::syllabic)
    }

    #[must_use]
    pub fn consonant(self) -> bool {
        self.features().map_or(false, Features::consonant)
    }

    #[allow(clippy::match_same_arms)]
    #[must_use]
    pub fn matcher_diff(self, pattern: Option<Self>) -> f32 {
        match (self, pattern) {
            (Element::Features(_), Some(Element::Other(_)))
            | (Element::Other(_), Some(Element::Features(_))) => 1.0,
            (Element::Features(l), Some(Element::Features(r))) => {
                l.syllabic_aware_diff(r).unwrap_or(1.0)
            },
            (Element::Other(l), Some(Element::Other(r))) if l == r => 0.0,
            (Element::Other(_), Some(Element::Other(_))) => 1.0,
            (_, None) => 0.0,
        }
    }

    #[must_use]
    pub fn stress_marker(self) -> Option<StressMarker> {
        match self {
            Element::Other('ˈ') => Some(StressMarker::Primary),
            Element::Other('ˌ') => Some(StressMarker::Secondary),
            Element::Other('.') => Some(StressMarker::None),
            Element::Features(_) | Element::Other(_) => None,
        }
    }
}

impl fmt::Display for Element {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Element::Features(x) => write!(f, "{x}"),
            Element::Other(c) => write!(f, "{c}"),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug)]
pub enum StressMarker {
    #[default]
    None,
    Secondary,
    Primary,
}

pub mod parser {
    use nom::{
        branch::alt,
        bytes::complete::tag,
        character::complete::anychar,
        combinator::{flat_map, map_opt, peek, rest},
        multi::fold_many0,
        sequence::pair,
        IResult, Parser,
    };

    #[allow(clippy::wildcard_imports)]
    use super::*;

    fn feature_map_elem(feature_map: &FeatureMap) -> impl Fn(&str) -> IResult<&str, Features> + '_ {
        move |s| {
            flat_map(
                map_opt(
                    pair(
                        map_opt(anychar::<&str, _>, |c| feature_map.get(&c)),
                        peek(rest),
                    ),
                    |(options, rest)| {
                        options
                            .iter()
                            .filter(|&&(option, _)| rest.starts_with(option))
                            .copied()
                            .next()
                    },
                ),
                |(option, out): (&str, Features)| tag(option).map(move |_| out),
            )
            .parse(s)
        }
    }

    pub fn parse_features(input: &str) -> IResult<&str, Features> {
        let feature_map = &*DATA;

        flat_map(
            pair(
                fold_many0(
                    feature_map_elem(&feature_map.prefixes),
                    || Features::NEUTRAL,
                    Features::overridden_with,
                ),
                feature_map_elem(&feature_map.bases),
            ),
            |(prefix, base)| {
                fold_many0(
                    feature_map_elem(&feature_map.suffixes),
                    move || base.overridden_with(prefix),
                    Features::overridden_with,
                )
            },
        )
        .parse(input)
    }

    pub fn parse_element(input: &str) -> IResult<&str, Element> {
        alt((
            parse_features.map(Element::Features),
            anychar.map(Element::Other),
        ))
        .parse(input)
    }
}

pub fn segments(mut input: &str) -> impl Iterator<Item = (&str, Element)> {
    use nom::{combinator::consumed, Finish, Parser};

    iter::from_fn(move || {
        if input.is_empty() {
            None
        } else {
            let (rest, out) = consumed(parser::parse_element)
                .parse(input)
                .finish()
                .unwrap(); // `anychar` cannot fail for non-empty inputs
            input = rest;
            Some(out)
        }
    })
}

#[must_use]
pub fn extract_syllables(input: &mut Vec<Element>) -> Vec<(usize, StressMarker)> {
    let mut markers = Vec::new();

    let mut idx = 0;
    while let Some(elem) = input.get(idx) {
        match elem.stress_marker() {
            None => idx += 1,
            Some(marker) => {
                input.remove(idx);
                if markers.last().map_or(true, |&(last_idx, _)| last_idx < idx) {
                    markers.push((idx, marker));
                }
            },
        }
    }

    markers
}
