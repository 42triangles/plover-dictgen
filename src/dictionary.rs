use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    hash::BuildHasher,
    mem,
};

use itertools::Itertools;

use crate::{
    system::{Arenas, PreparedWord, System},
    Weight,
};

pub struct Item {
    pub word: String,
    pub translations: Vec<(Weight, String)>,
}

#[must_use]
pub fn create_item(
    system: &System,
    word: &PreparedWord,
    take: usize,
    max_size: Option<usize>,
) -> Item {
    let mut translations = system
        .possible_translations(word, max_size, &Arenas::default())
        .map(|(weight, candidate)| {
            (
                weight,
                candidate
                    .translation
                    .into_iter()
                    .map(|i| i.string)
                    .join("/"),
            )
        })
        .take(take)
        .collect::<Vec<_>>();

    translations.reverse();

    Item {
        word: word.ortho.clone(),
        translations,
    }
}

#[derive(Clone)]
struct WordCandidate<'a> {
    word: &'a str,
    translations: Vec<(Weight, Weight, &'a str)>,
}

#[must_use]
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn build(
    items: &[Item],
    word_counts: &HashMap<String, usize, impl BuildHasher>,
    blocked: &HashSet<String, impl BuildHasher>,
) -> HashMap<String, String> {
    let word_count =
        |i: &str| Weight::new(word_counts.get(i).copied().unwrap_or_default() as f32).unwrap();

    let mut candidates = items
        .iter()
        .map(|item| {
            let adjust =
                Weight::new((word_count(&item.word) + Weight::new(1.0).unwrap()).ln()).unwrap();
            let mut last_weight = None;
            let mut translations = item
                .translations
                .iter()
                .filter(|&&(_, ref translation)| !blocked.contains(translation))
                .map(|&(weight, ref translation)| {
                    (
                        mem::replace(&mut last_weight, Some(weight))
                            .map_or_else(Weight::default, |w| weight - w)
                            * adjust,
                        weight * adjust,
                        &**translation,
                    )
                })
                .collect::<Vec<_>>();
            translations.reverse();
            WordCandidate {
                word: &item.word,
                translations,
            }
        })
        .collect::<Vec<_>>();

    candidates.sort_by_key(|i| word_count(i.word));

    let mut dictionary_progress = HashMap::<&str, WordCandidate>::default();

    for entry in &candidates {
        let mut entry = entry.clone();
        loop {
            match dictionary_progress.entry(entry.translations.pop().unwrap().2) {
                Entry::Vacant(vacant) => {
                    vacant.insert(entry);
                    break;
                },
                Entry::Occupied(mut occupied) => {
                    let displace = match (
                        entry.translations.last(),
                        occupied.get().translations.last(),
                    ) {
                        (None, None) => {
                            eprintln!(
                                "Can't handle conflict between '{}' and '{}'",
                                occupied.get().word,
                                entry.word
                            );
                            if word_count(entry.word) > word_count(occupied.get().word) {
                                mem::swap(&mut entry, occupied.get_mut());
                            }
                            eprintln!("'{}' is not present in the final dictionary", entry.word);
                            break;
                        },
                        (None, Some(_)) => true,
                        (Some(_), None) => false,
                        (Some(l), Some(r)) => l.0 >= r.0,
                    };

                    if displace {
                        mem::swap(&mut entry, occupied.get_mut());
                    }

                    continue;
                },
            }
        }
    }

    let mut filler = HashMap::<&str, (Weight, &str, bool)>::default();

    for i in &candidates {
        for j in &i.translations {
            if !dictionary_progress.contains_key(&j.2) {
                match filler.entry(j.2) {
                    Entry::Vacant(vacant) => {
                        vacant.insert((j.1, i.word, false));
                    },
                    Entry::Occupied(mut occupied) => {
                        if occupied.get().0 > j.1 {
                            occupied.insert((j.1, i.word, false));
                        }
                    },
                }
            }
        }
    }

    for i in &candidates {
        for j in &i.translations {
            for slash_idx in j.2.match_indices('/').map(|(idx, _)| idx) {
                let new_translation = &j.2[..slash_idx];

                if dictionary_progress.contains_key(new_translation) {
                    continue;
                }

                match filler.entry(j.2) {
                    Entry::Vacant(vacant) => {
                        vacant.insert((j.1, i.word, true));
                    },
                    Entry::Occupied(mut occupied) => {
                        let (occupied_weight, _, true) = *occupied.get() else { continue };
                        if occupied_weight > j.1 {
                            occupied.insert((j.1, i.word, true));
                        }
                    },
                }
            }
        }
    }

    dictionary_progress
        .into_iter()
        .map(|(key, value)| (key, value.word))
        .chain(filler.into_iter().map(|(key, (_, value, _))| (key, value)))
        .map(|(key, value)| (key.to_owned(), value.to_owned()))
        .collect()
}
