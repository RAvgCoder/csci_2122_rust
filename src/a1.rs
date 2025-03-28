use crate::helpers::read_line;

pub struct Assignment1;

impl Assignment1 {
    pub fn spellbee() {
        let words = Self::innit_word_list();
        for word in &words {
            if word.is_panagram() {
                let words_matched: usize = words
                    .iter()
                    .map(|other_word| {
                        (other_word.char_set & word.char_set == other_word.char_set) as usize
                    })
                    .sum();

                println!(
                    "{} : {} ; {}",
                    word.word,
                    word.get_visible_char_set(),
                    words_matched
                );
            }
        }
    }

    fn innit_word_list() -> Box<[Word]> {
        let mut std_in = std::io::stdin();
        let number_of_words = read_line(&mut std_in)
            .expect("Error reading input")
            .trim()
            .parse::<usize>()
            .expect("Error parsing input");

        (0..number_of_words)
            .map(|_| {
                let word = read_line(&mut std_in)
                    .expect("Error reading input")
                    .trim()
                    .to_string();

                Word::new(word)
            })
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Word {
    word: String,
    // The bits represent which of the characters from A-Z exists.
    // bits are stored in reverse so that the least significant bit represents 'A'.
    // Eg: [_____EDCBA]
    char_set: u32,
}

impl Word {
    fn new(word: String) -> Self {
        let mut char_set = 0;
        for c in word.chars() {
            let c = c as u8 - b'A';
            char_set |= 1 << c;
        }
        Self { word, char_set }
    }

    fn is_panagram(&self) -> bool {
        const PANAGRAM_LEN: u32 = 7;
        self.char_set.count_ones() == PANAGRAM_LEN
    }

    fn get_visible_char_set(&self) -> String {
        let mut char_map = String::with_capacity(26);
        for i in 0..26 {
            if self.char_set & (1 << i) != 0 {
                char_map.push((i + b'A') as char);
                char_map.push(' ');
            }
        }
        char_map.pop();
        char_map
    }
}
