#![allow(unused_crate_dependencies)]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use jemallocator::Jemalloc;
use plover_dictgen::{ipa, matcher::MixedInput, system::System};

#[global_allocator]
static ALLOC: Jemalloc = Jemalloc;

const EXAMPLES: &[(&str, &str)] = &[
    ("", ""),
    ("stɪˈnɒɡɹəfi", "stenography"),
    ("ˈbɛntʃmɑɹk", "benchmark"),
    ("pɚ.ˈfɔɹ.məns", "performance"),
    ("ˈdaɪkstɹə", "dijkstra"),
    (
        "ˌsupɚˌkælɪˌfɹædʒɪˌlɪstɪkˌɛkspiˌælɪˈdoʊʃəs",
        "supercalifragilisticexpialidocious",
    ),
];

fn criterion_benchmarks(c: &mut Criterion) {
    ipa::load_data_early();

    let config = System::load("english-plover.toml");
    let language = config.language();

    for (ipa, ortho) in EXAMPLES.iter().copied() {
        c.bench_with_input(BenchmarkId::new("substitutions", ortho), &ipa, |b, &ipa| {
            b.iter(|| language.ipa_with_substitutions(ipa.to_owned()))
        });
        let mut ipa = language.ipa_with_substitutions(ipa.to_owned());

        c.bench_with_input(
            BenchmarkId::new("extract_syllables", ortho),
            &&*ipa,
            |b, &ipa| b.iter(|| ipa::extract_syllables(&mut ipa.to_owned())),
        );
        let stress_markers = ipa::extract_syllables(&mut ipa);

        c.bench_with_input(
            BenchmarkId::new("map_ortho", ortho),
            &(&*ipa, ortho),
            |b, &(ipa, ortho)| b.iter(|| language.map_ortho(ipa, ortho)),
        );
        let mapping = language.map_ortho(&ipa, ortho);

        c.bench_with_input(
            BenchmarkId::new("split_syllables", ortho),
            &(&*stress_markers, &*ipa, ortho, &*mapping),
            |b, &(stress_markers, ipa, ortho, mapping)| {
                b.iter(|| {
                    language.split_syllables(&mut stress_markers.to_owned(), MixedInput {
                        left: ipa,
                        right: ortho,
                        assoc: mapping,
                    })
                })
            },
        );
    }

    for (ipa, ortho) in EXAMPLES[..EXAMPLES.len() - 1].iter().copied() {
        c.bench_with_input(
            BenchmarkId::new("full", ortho),
            &(ipa, ortho),
            |b, &(ipa, ortho)| {
                b.iter(|| {
                    let word = config.prepare_ipa(ipa, ortho);
                    let out = config.possible_translations(&word, None, &Default::default()).next();
                    out
                })
            },
        );
    }
}

criterion_group!(benches, criterion_benchmarks);
criterion_main!(benches);
