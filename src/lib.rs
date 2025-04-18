#![feature(coroutines, coroutine_clone, coroutine_trait, stmt_expr_attributes)]
#![allow(
    clippy::if_not_else,
    clippy::needless_borrowed_reference,
    clippy::redundant_else,
    clippy::match_on_vec_items,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

pub mod arpabet;
#[macro_use]
pub mod explore;
pub mod dictionary;
pub mod helpers;
pub mod ipa;
pub mod language;
pub mod matcher;
pub mod system;

use educe::Educe;
use helpers::TomlOption;
use jemallocator as _; // used in the binary
use matcher::Costs;
use ordered_float::NotNan;
use serde::Deserialize;
use serde_json as _; // used in the binary

type Weight = NotNan<f32>;

fn ipa_pow_default() -> Weight {
    Weight::new(0.2).unwrap()
}

#[allow(clippy::unnecessary_wraps)] // for `serde(default)`
fn ipa_weight_default() -> TomlOption<Weight> {
    TomlOption(Some(Weight::new(1.0).unwrap()))
}

fn ipa_change_default() -> bool {
    true
}

#[derive(Clone, Copy, Deserialize, Debug, Educe)]
#[educe(Default)]
struct IpaWeights {
    #[educe(Default(expression = "ipa_pow_default()"))]
    #[serde(default = "ipa_pow_default")]
    pow: Weight,
    #[educe(Default(expression = "ipa_weight_default()"))]
    #[serde(default = "ipa_weight_default")]
    ignored: TomlOption<Weight>,
    #[educe(Default(expression = "ipa_weight_default()"))]
    #[serde(default = "ipa_weight_default")]
    unmatched: TomlOption<Weight>,
    #[educe(Default(expression = "ipa_change_default()"))]
    #[serde(default = "ipa_change_default")]
    allow_change: bool,
}

impl IpaWeights {
    fn costs(self) -> Costs<Weight> {
        Costs {
            ignored: self.ignored.0,
            unmatched: self.unmatched.0,
        }
    }

    fn diff(self, l: Option<ipa::Element>, r: ipa::Element) -> Option<Weight> {
        if self.allow_change {
            Some(NotNan::new(r.matcher_diff(l).powf(self.pow.into_inner())).unwrap())
        } else if l.map_or(true, |l| l == r) {
            Some(Weight::default())
        } else {
            None
        }
    }
}

fn ortho_scale_default() -> bool {
    true
}

#[allow(clippy::unnecessary_wraps)] // for `serde(default)`
fn ortho_ignore_default() -> TomlOption<Weight> {
    TomlOption(Some(Weight::new(0.01).unwrap()))
}

#[allow(clippy::unnecessary_wraps)] // for `serde(default)`
fn ortho_weight_default() -> TomlOption<Weight> {
    TomlOption(Some(Weight::new(1.0).unwrap()))
}

#[derive(Clone, Copy, Deserialize, Debug, Educe)]
#[educe(Default)]
struct OrthoWeights {
    #[educe(Default(expression = "ortho_scale_default()"))]
    #[serde(default = "ortho_scale_default")]
    scale: bool,
    #[educe(Default(expression = "ortho_ignore_default()"))]
    #[serde(default = "ortho_ignore_default")]
    ignored: TomlOption<Weight>,
    #[educe(Default(expression = "ortho_weight_default()"))]
    #[serde(default = "ortho_weight_default")]
    unmatched: TomlOption<Weight>,
    #[educe(Default(expression = "ortho_weight_default()"))]
    #[serde(default = "ortho_weight_default")]
    changed: TomlOption<Weight>,
}

impl OrthoWeights {
    fn costs(self, factor: Weight) -> Costs<Weight> {
        Costs {
            ignored: self.ignored.0.map(|x| x * factor),
            unmatched: self.unmatched.0.map(|x| x * factor),
        }
    }

    fn diff_inner(l: Option<char>, r: char, default: Option<Weight>) -> Option<Weight> {
        if l.map_or(true, |l| l == r) {
            Some(Weight::default())
        } else {
            default
        }
    }

    fn diff(self, l: Option<char>, r: char, factor: Weight) -> Option<Weight> {
        Self::diff_inner(l, r, self.changed.0).map(|out| out * factor)
    }

    fn diff1(l: Option<char>, r: char) -> Option<Weight> {
        Self::diff_inner(l, r, Some(Weight::new(1.0).unwrap()))
    }

    #[allow(clippy::cast_precision_loss)]
    fn factor(self, ipa_len: usize, ortho_len: usize) -> Weight {
        if self.scale {
            Weight::new(ipa_len as f32 / ortho_len.max(1) as f32).unwrap()
        } else {
            Weight::new(1.0).unwrap()
        }
    }
}
