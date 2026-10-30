//! Word error rate (WER): the standard speech-recognition distance between a
//! reference transcript and a hypothesis.
//!
//! WER = (substitutions + insertions + deletions) / reference words, where the
//! three counts come from the minimum-edit alignment of the two WORD sequences
//! (Levenshtein distance over words, not characters). Both texts are
//! normalised first (lower-case, punctuation removed, whitespace collapsed) so
//! that "Hello, world!" and "hello world" are the same utterance.
//!
//! Pure functions, no I/O. `wer_corpus` pools the counts over many pairs
//! (sum of errors / sum of reference words), which is the conventional corpus
//! WER and weights long clips more than a mean of per-clip rates would.

/// Alignment counts between one reference and one hypothesis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WerCounts {
    pub reference_words: usize,
    pub hypothesis_words: usize,
    pub substitutions: usize,
    pub insertions: usize,
    pub deletions: usize,
}

impl WerCounts {
    pub fn errors(&self) -> usize {
        self.substitutions + self.insertions + self.deletions
    }

    /// Errors per reference word. `None` when the reference is empty but the
    /// hypothesis is not (the ratio is undefined); an empty reference with an
    /// empty hypothesis is a perfect match, 0.0.
    pub fn wer(&self) -> Option<f64> {
        if self.reference_words == 0 {
            return if self.errors() == 0 { Some(0.0) } else { None };
        }
        Some(self.errors() as f64 / self.reference_words as f64)
    }

    pub fn add(&mut self, other: &WerCounts) {
        self.reference_words += other.reference_words;
        self.hypothesis_words += other.hypothesis_words;
        self.substitutions += other.substitutions;
        self.insertions += other.insertions;
        self.deletions += other.deletions;
    }
}

/// Lower-case, drop punctuation, collapse whitespace; returns the word list.
/// Apostrophes inside a word are kept (`don't` stays one word); every other
/// non-alphanumeric character is a separator.
pub fn normalise_words(text: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut current = String::new();
    for ch in text.chars().flat_map(char::to_lowercase) {
        if ch.is_alphanumeric() || ch == '\'' || ch == '\u{2019}' {
            current.push(if ch == '\u{2019}' { '\'' } else { ch });
        } else if !current.is_empty() {
            words.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        words.push(current);
    }
    words
        .into_iter()
        .map(|w| w.trim_matches('\'').to_owned())
        .filter(|w| !w.is_empty())
        .collect()
}

/// Minimum-edit alignment of two word sequences with the error breakdown.
/// Ties are resolved deterministically: substitution, then deletion, then
/// insertion.
pub fn align_words(reference: &[String], hypothesis: &[String]) -> WerCounts {
    let (n, m) = (reference.len(), hypothesis.len());
    // cell = (distance, substitutions, insertions, deletions)
    let mut dp = vec![vec![(0usize, 0usize, 0usize, 0usize); m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate().skip(1) {
        row[0] = (i, 0, 0, i);
    }
    for j in 1..=m {
        dp[0][j] = (j, 0, j, 0);
    }
    for i in 1..=n {
        for j in 1..=m {
            let same = reference[i - 1] == hypothesis[j - 1];
            let diag = dp[i - 1][j - 1];
            let sub = if same {
                diag
            } else {
                (diag.0 + 1, diag.1 + 1, diag.2, diag.3)
            };
            let up = dp[i - 1][j];
            let del = (up.0 + 1, up.1, up.2, up.3 + 1);
            let left = dp[i][j - 1];
            let ins = (left.0 + 1, left.1, left.2 + 1, left.3);
            let mut best = sub;
            if del.0 < best.0 {
                best = del;
            }
            if ins.0 < best.0 {
                best = ins;
            }
            dp[i][j] = best;
        }
    }
    let (_, substitutions, insertions, deletions) = dp[n][m];
    WerCounts {
        reference_words: n,
        hypothesis_words: m,
        substitutions,
        insertions,
        deletions,
    }
}

/// WER counts of `hypothesis` against `reference`, both normalised.
pub fn word_error(reference: &str, hypothesis: &str) -> WerCounts {
    align_words(&normalise_words(reference), &normalise_words(hypothesis))
}

/// Pooled counts over many (reference, hypothesis) pairs: corpus WER is
/// `errors()` / `reference_words` of the result.
pub fn wer_corpus<'a, I>(pairs: I) -> WerCounts
where
    I: IntoIterator<Item = (&'a str, &'a str)>,
{
    let mut total = WerCounts::default();
    for (reference, hypothesis) in pairs {
        total.add(&word_error(reference, hypothesis));
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-12
    }

    #[test]
    fn identical_text_is_zero() {
        let c = word_error("the quick brown fox jumps", "the quick brown fox jumps");
        assert_eq!(c.errors(), 0);
        assert_eq!(c.reference_words, 5);
        assert_eq!(c.wer(), Some(0.0));
    }

    #[test]
    fn one_substitution_in_five_words_is_point_two() {
        let c = word_error("the quick brown fox jumps", "the quick brown cat jumps");
        assert_eq!(c.substitutions, 1);
        assert_eq!(c.insertions, 0);
        assert_eq!(c.deletions, 0);
        assert!(close(c.wer().unwrap(), 0.2));
    }

    #[test]
    fn insertions_and_deletions_are_counted_separately() {
        let del = word_error("one two three four", "one three four");
        assert_eq!((del.substitutions, del.insertions, del.deletions), (0, 0, 1));
        assert!(close(del.wer().unwrap(), 0.25));
        let ins = word_error("one two three four", "one two extra three four");
        assert_eq!((ins.substitutions, ins.insertions, ins.deletions), (0, 1, 0));
        assert!(close(ins.wer().unwrap(), 0.25));
        // Everything deleted: WER 1.0. Everything doubled: WER can exceed 1.0.
        assert!(close(word_error("a b c", "").wer().unwrap(), 1.0));
        assert!(close(word_error("a", "x y z").wer().unwrap(), 3.0));
    }

    #[test]
    fn empty_reference_is_handled() {
        assert_eq!(word_error("", "").wer(), Some(0.0));
        let c = word_error("", "unexpected words");
        assert_eq!(c.insertions, 2);
        assert_eq!(c.wer(), None);
        assert_eq!(word_error("...", "").wer(), Some(0.0), "punctuation-only reference normalises to empty");
    }

    #[test]
    fn punctuation_and_case_are_ignored() {
        let c = word_error("Hello, World!  It's  fine.", "hello world it's fine");
        assert_eq!(c.errors(), 0);
        assert_eq!(normalise_words("Don\u{2019}t stop -- 'quoted' word."), vec!["don't", "stop", "quoted", "word"]);
        assert_eq!(normalise_words("  \t\n "), Vec::<String>::new());
    }

    #[test]
    fn corpus_pools_counts_not_rates() {
        // Clip A: 1 error in 1 word (rate 1.0); clip B: 0 errors in 9 words.
        // Mean of rates would be 0.5; corpus WER is 1/10.
        let total = wer_corpus([("yes", "no"), ("a b c d e f g h i", "a b c d e f g h i")]);
        assert_eq!(total.reference_words, 10);
        assert_eq!(total.errors(), 1);
        assert!(close(total.wer().unwrap(), 0.1));
        assert_eq!(wer_corpus(Vec::<(&str, &str)>::new()).wer(), Some(0.0));
    }
}
