#![allow(unused_crate_dependencies)] // they're used in the `lib.rs`
#![allow(
    clippy::if_not_else,
    clippy::needless_borrowed_reference,
    clippy::redundant_else,
    clippy::match_on_vec_items,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc
)]

use jemallocator::Jemalloc;

#[global_allocator]
static ALLOC: Jemalloc = Jemalloc;

use std::env::args;

use itertools::Itertools;
use plover_dictgen::{
    helpers::parse_size,
    ipa,
    system::{Arenas, System},
};
use serde::Serialize;

#[derive(Serialize)]
struct Candidate {
    weight: f32,
    translation: String,
    rule_history: Vec<String>,
}

#[allow(clippy::use_debug)]
fn main() {
    ipa::load_data_early();

    let config = System::load(args().nth(1).unwrap());

    let word = config.prepare_ipa(&args().nth(2).unwrap(), &args().nth(3).unwrap());

    #[cfg(debug_assertions)]
    {
        println!("{word}");
    }

    let out = config
        .possible_translations(
            &word,
            args().nth(4).and_then(|s| parse_size(&s)),
            &Arenas::default(),
        )
        .map(|(weight, candidate)| Candidate {
            weight: weight.into(),
            translation: candidate.translation.iter().map(|i| &i.string).join("/"),
            rule_history: candidate
                .rule_history
                .iter()
                .map(|i| i.string.clone())
                .collect(),
        })
        .take(if cfg!(debug_assertions) { 16 } else { 128 })
        .collect::<Vec<_>>();

    #[cfg(debug_assertions)]
    {
        println!("{}", serde_json::to_string_pretty(&out).unwrap());
    }

    #[cfg(not(debug_assertions))]
    {
        println!("{}", serde_json::to_string(&out).unwrap());
    }
}
