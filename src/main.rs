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

#[allow(
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation
)]
fn parse_size(mut s: &str) -> Option<usize> {
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

#[allow(clippy::use_debug)]
fn main() {
    ipa::load_data_early();

    let config = System::load(args().nth(1).unwrap());

    /*
    println!(
        "{}",
        config.prepare_arpa(
            "S UW2 P ER0 K AE2 L AH0 F R AE1 JH AH0 L IH2 S T IH0 K EH2 K S P IY0 AE2 L AH0 D OW1 \
             SH AH0 S",
            "supercalifragilisticexpialidocious"
        )
    );
    */
    //  let word = config.prepare_arpa(
    //      "S UW2 P ER0 K AE2 L AH0 F R AE1 JH AH0 L IH2 S T IH0 K EH2 K S P IY0
    // AE2 L AH0 D OW1 \       SH AH0 S",
    //      "supercalifragilisticexpialidocious"
    //  );

    //let word = config.prepare_ipa("ˌsupɚˌkælɪˌfɹædʒɪˌlɪstɪk",
    // "supercalifragilistic");
    let word = config.prepare_arpa(&args().nth(2).unwrap(), &args().nth(3).unwrap());

    //let word = config.prepare_ipa("stəˈnɑɡɹəfi", "stenography");
    //let word = config.prepare_ipa("ˈbɛntʃmɑɹk", "benchmark");
    //let word = config.prepare_ipa("pɚˈfɔɹməns", "performance");
    //let word = config.prepare_ipa("pɜː(ɹ)ˈtʃɑːns", "perchance");
    //let word = config.prepare_ipa("fɛsts", "fests");
    //let word = config.prepare_ipa("ˈtʃɑːns", "chance");
    //let word = config.prepare_arpa("D IH1 K SH AH0 N", "diction");
    //let word = config.prepare_arpa("D IH1 K T", "dict");
    //let word = config.prepare_ipa("fɛsts", "fests");
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
