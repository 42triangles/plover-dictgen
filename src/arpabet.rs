use crate::ipa::{self, StressMarker};

#[derive(Default)]
pub struct ToIpa {
    pub ipa: Vec<ipa::Element>,
    pub stress_markers: Vec<(usize, StressMarker)>,
    pub vowel_stress: Vec<(usize, StressMarker)>,
}

const VOWELS: &[(&str, &str)] = &[
    ("AA", "ɑ"),
    ("AE", "æ"),
    ("AH0", "ə"),
    ("AH", "ʌ"),
    ("AO", "ɔ"),
    ("AW", "aʊ"),
    ("AX", "ə"),
    ("AXR", "əɹ"),
    ("AY", "aɪ"),
    ("EH", "ɛ"),
    ("ER0", "əɹ"),
    ("ER", "ɜɹ"),
    ("EY", "eɪ"),
    ("IH", "ɪ"),
    ("IX", "ɨ"),
    ("IY", "i"),
    ("OW", "oʊ"),
    ("OY", "ɔɪ"),
    ("UH", "ʊ"),
    ("UW", "u"),
    ("UX", "ʉ"),
];

const CONSONANTS: &[(&str, &str)] = &[
    ("B", "b"),
    ("CH", "t̠ʃ"),
    ("D", "d"),
    ("DH", "ð"),
    ("DX", "ɾ"),
    ("EL", "l̩"),
    ("EM", "m̩"),
    ("EN", "n̩"),
    ("F", "f"),
    ("G", "ɡ"),
    ("HH", "h"),
    ("H", "h"),
    ("JH", "d̠ʒ"),
    ("K", "k"),
    ("L", "l"),
    ("M", "m"),
    ("N", "n"),
    ("NG", "ŋ"),
    ("NX", "ɾ̃"),
    ("P", "p"),
    ("Q", "ʔ"),
    ("R", "ɹ"),
    ("S", "s"),
    ("SH", "ʃ"),
    ("T", "t"),
    ("TH", "θ"),
    ("V", "v"),
    ("W", "w"),
    ("WH", "ʍ"),
    ("Y", "j"),
    ("Z", "z"),
    ("ZH", "ʒ"),
];

#[must_use]
pub fn to_ipa(s: &str) -> ToIpa {
    let mut out = ToIpa::default();

    let segments = |out: &mut ToIpa, ipa: &str| out.ipa.extend(ipa::segments(ipa).map(|(_, x)| x));

    let mut last_was_vowel = true;
    'outer: for i in s.split_ascii_whitespace() {
        for &(consonant, ipa) in CONSONANTS {
            if i == consonant {
                segments(&mut out, ipa);
                last_was_vowel = false;
                continue 'outer;
            }
        }

        for &(vowel, ipa) in VOWELS {
            if let Some(rest) = i.strip_prefix(vowel) {
                let stress = match rest {
                    "" | "0" => StressMarker::None,
                    "1" => StressMarker::Primary,
                    "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" => StressMarker::Secondary,
                    s if !s.chars().next().unwrap().is_alphabetic() => StressMarker::None,
                    _ => continue,
                };

                let marker = (out.ipa.len(), stress);

                segments(&mut out, ipa);

                out.vowel_stress.push(marker);
                if last_was_vowel {
                    out.stress_markers.push(marker);
                }

                last_was_vowel = true;
                continue 'outer;
            }
        }
    }

    out
}
