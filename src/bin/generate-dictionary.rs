#![feature(is_sorted)]
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

use std::{
    env::args,
    fs::{self, File},
    io::{prelude::*, BufReader},
};

use ordered_float::NotNan;
use plover_dictgen::{dictionary::{self, Item}, helpers::UntaggedEither};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::Deserialize;

#[derive(Clone, Deserialize)]
struct StenoCandidate {
    weight: NotNan<f32>,
    translation: String,
}

#[derive(Clone, Deserialize)]
struct InputEntry {
    word: String,
    translations: Vec<StenoCandidate>,
}

#[allow(clippy::use_debug)]
fn main() {
    let mut word_counts = HashMap::<String, usize>::default();

    for i in BufReader::new(File::open(args().nth(2).unwrap()).unwrap()).lines() {
        let (word, count) = i.as_ref().unwrap().rsplit_once(' ').unwrap();
        *word_counts.entry(word.to_lowercase()).or_insert(0) += count.parse::<usize>().unwrap();
    }

    let items = BufReader::new(File::open(args().nth(1).unwrap()).unwrap())
        .lines()
        .map(|i| {
            let entry = serde_json::from_str::<InputEntry>(&i.unwrap()).unwrap();
            Item {
                word: entry.word,
                translations: entry
                    .translations
                    .into_iter()
                    .map(|i| (i.weight, i.translation))
                    .collect(),
            }
        })
        .collect::<Vec<_>>();

    let mut blocked = HashSet::default();

    for path in args().skip(3) {
        type Blocking = UntaggedEither<Vec<String>, HashMap<String, serde_json::Value>>;
        match serde_json::from_str::<Blocking>(&fs::read_to_string(path).unwrap()).unwrap() {
            UntaggedEither::Left(set) => blocked.extend(set),
            UntaggedEither::Right(map) => blocked.extend(map.into_keys()),
        };
    }

    let dictionary = dictionary::build(&items, &word_counts, &blocked);

    println!("{}", serde_json::to_string(&dictionary).unwrap());
}
