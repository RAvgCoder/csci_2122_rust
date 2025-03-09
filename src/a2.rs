use crate::a2::stream_core::{BitStreamReader, ConsoleReader, StreamReader};
use std::num::NonZeroU8;

pub struct Assignment2;

#[derive(Eq, PartialEq, Debug, Clone)]
struct CallBackResult {
    line: u32,
    integer: u32,
    bit_seq: i32,
}

impl Assignment2 {
    #[allow(dead_code)]
    pub fn marsdecx() {
        let mut reader = ConsoleReader::new();
        Self::marsdec_core::<true>(&mut reader, |result| {
            println!(
                "Line {}, integer {}: {}",
                result.line, result.integer, result.bit_seq
            );
        });
    }

    #[allow(dead_code)]
    pub fn marsdec1() {
        let mut reader = ConsoleReader::new();
        Self::marsdec_core::<false>(&mut reader, |result| {
            println!(
                "Line {}, integer {}: {}",
                result.line, result.integer, result.bit_seq
            );
        });
    }

    fn marsdec_core<const DECODE_MULTIPLE: bool>(
        reader: &mut impl StreamReader,
        mut call_back: impl FnMut(CallBackResult),
    ) {
        for i in 1.. {
            // Get the next stream else if there are no more streams, then exit
            let Some(mut hex_stream) = reader.next_stream() else {
                break;
            };

            let mut bit_stream_reader = BitStreamReader::new(&mut hex_stream);
            for j in 1.. {
                // Read the first 5 bits to get the size of the next bit sequence to read
                const HEADER_SIZE: NonZeroU8 = match NonZeroU8::new(5) {
                    None => panic!("Expected a non-zero value"),
                    Some(val) => val,
                };

                let Some(bit_seq) = bit_stream_reader.take(HEADER_SIZE) else {
                    break;
                };

                // If the first 5 bits are 0, then we have reached the end of the stream
                let Some(first_5_bits_value) = NonZeroU8::new(bit_seq.as_raw() as u8) else {
                    break;
                };

                let Some(bit_seq) = bit_stream_reader.take(first_5_bits_value) else {
                    break;
                };

                let result: i32 = bit_seq.using_twos_complement();

                call_back(CallBackResult {
                    line: i,
                    integer: j,
                    bit_seq: result,
                });

                if !DECODE_MULTIPLE {
                    break;
                }
            }
        }
    }
}

mod stream_core {
    use crate::helpers::make_one_bits;
    use crate::helpers::size_of_in_bits;
    use std::fmt::Debug;
    use std::num::NonZeroU8;

    /// A sequence of bits which are right aligned
    pub struct BitSequence {
        seq: u32,
        len: NonZeroU8,
    }

    impl Debug for BitSequence {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "BitSeq {{ len: {} seq: ", self.len)?;
            for i in (0..self.len.get()).rev() {
                write!(f, "{}", (self.seq >> i) & 1)?;
            }
            write!(f, " }}")
        }
    }

    impl BitSequence {
        pub const fn new(seq: u32, len: NonZeroU8) -> Self {
            BitSequence { seq, len }
        }

        pub const fn as_raw(&self) -> u32 {
            self.seq
        }

        pub const fn using_twos_complement(&self) -> i32 {
            let is_first_bit_set = ((self.seq >> (self.len.get() - 1)) & 1) == 1;
            let raw_num = self.seq;
            if is_first_bit_set {
                // Use twos complement
                (raw_num | (-1 << self.len.get()) as u32) as i32
            } else {
                raw_num as i32
            }
        }
    }

    pub struct BitStreamReader<'stream, S> {
        /// The stream that provides the bytes to be read
        byte_stream: &'stream mut S,
        /// The cache that we would use to store the bits that we have read
        cache: u64,
        /// The number of bits in the cache
        cache_size: u8,
    }

    impl<'stream, S: ByteStream> BitStreamReader<'stream, S> {
        /// The maximum number of bits that can be stored in the cache
        const MAX_CACHE_SIZE: u8 = size_of_in_bits::<u64>() as u8;

        pub fn new(byte_stream: &'stream mut S) -> Self {
            BitStreamReader {
                byte_stream,
                cache: 0,
                cache_size: 0,
            }
        }

        pub fn take(&mut self, num_bits: NonZeroU8) -> Option<BitSequence> {
            assert!(num_bits.get() <= 32, "Cannot take more than 32 bits");

            if self.cache_size < num_bits.get() {
                let num_bits_to_read = num_bits.get() - self.cache_size;
                // Round up to the nearest multiple of 8
                let num_bits_to_read = (num_bits_to_read + 7) & !7;
                let num_bytes_to_read = num_bits_to_read / 8;

                // Read the bytes
                // NOTE: Bits are stored right aligned 2 is stored as 0000_0000_0000_0010
                let mut bytes_accumulated: u32 = 0;
                for _ in 0..num_bytes_to_read {
                    let byte: u8 = self.byte_stream.next_byte()?;
                    bytes_accumulated <<= 8;
                    bytes_accumulated |= byte as u32;
                }

                let shift_by = {
                    let start_offset = Self::MAX_CACHE_SIZE - self.cache_size;
                    (start_offset - num_bits_to_read) as u64
                };

                // Add the bits to the cache and update the cache size
                self.cache |= (bytes_accumulated as u64) << shift_by;
                self.cache_size += num_bits_to_read;
            }

            // SAFETY: We have already checked that the number of bits is at most 32
            let result_mask = unsafe { make_one_bits(num_bits.get()) };

            // Get the result from the cache and push the bits to the right so that the bits are right aligned
            // Eg For Cache: 1111_0000_0000_0000_0000_0000_0000_0000 => num_bits = 4
            // We get a result to be Result: 0000_0000_0000_0000_0000_0000_0000_1111 => num_bits = 4
            let result = (self.cache >> (Self::MAX_CACHE_SIZE - num_bits.get())) & result_mask;

            // SAFETY for cast: We have already checked that the number of bits is less than or equal to 32
            let result = BitSequence::new(result as u32, num_bits);

            // Update the cache
            self.cache <<= num_bits.get();
            self.cache_size -= num_bits.get();

            Some(result)
        }
    }

    pub trait ByteStream {
        fn next_byte(&mut self) -> Option<u8>;
    }

    #[repr(transparent)]
    pub struct HexStream<'reader>(std::str::SplitWhitespace<'reader>);

    impl HexStream<'_> {
        #[inline(always)]
        pub fn new(iter: std::str::SplitWhitespace) -> HexStream {
            HexStream(iter)
        }
    }

    impl ByteStream for HexStream<'_> {
        fn next_byte(&mut self) -> Option<u8> {
            self.0.next().map(|hex| {
                let stripped_hex = hex.strip_prefix("0x").unwrap_or(hex);
                u8::from_str_radix(stripped_hex, 16).expect("Expected a valid hex byte")
            })
        }
    }

    /// A trait representing a reader that provides streams of bytes.
    ///
    /// # Associated Types
    ///
    /// * `BStream<'stream>`: A type that implements the `ByteStream` trait.
    ///     This type is associated with a lifetime `'stream` which ensures that
    ///     the byte stream lives at least as long as the reader.
    ///
    /// # Required Methods
    ///
    /// * `next_stream(&mut self) -> Option<Self::BStream<'_>>`: This method should return the
    ///     next byte stream if available, or `None` if there are no more streams to read.
    pub trait StreamReader {
        type BStream<'stream>: ByteStream
        where
            Self: 'stream;
        fn next_stream(&mut self) -> Option<Self::BStream<'_>>;
    }

    /// Reads byte stream, line by line from std-in
    ///
    /// # Format
    /// ```
    /// 4
    /// 2 0x08 0x30
    /// 4 0x12 0x28 0x6c 0x70
    /// 8 0x43 0xdb 0x13 0x4b 0x2b 0xc6 0x14 0xe0
    /// ```
    pub struct ConsoleReader {
        num_of_streams_left: u32,
        curr_stream_processing: Option<String>,
    }

    impl ConsoleReader {
        fn read_line() -> String {
            let mut buffer = String::new();
            std::io::stdin()
                .read_line(&mut buffer)
                .expect("Expected to receive a byte stream");
            buffer
        }

        pub fn new() -> Self {
            ConsoleReader {
                num_of_streams_left: Self::read_line()
                    .trim()
                    .parse()
                    .expect("Expected a valid number"),
                curr_stream_processing: None,
            }
        }
    }

    impl StreamReader for ConsoleReader {
        type BStream<'reader> = HexStream<'reader>;

        /// 0x12 0x28 0x6c 0x70
        fn next_stream(&mut self) -> Option<Self::BStream<'_>> {
            if self.num_of_streams_left == 0 {
                return None;
            }

            let curr_stream = Self::read_line();

            self.curr_stream_processing = Some(curr_stream);
            self.num_of_streams_left -= 1;

            let mut iter = self
                .curr_stream_processing
                .as_ref()
                .unwrap()
                .split_whitespace();

            let _ = iter.next();

            Some(HexStream::new(iter))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a2::stream_core::{BitSequence, HexStream};
    use crate::helpers::{make_one_bits, size_of_in_bits};
    use rand::rngs::StdRng;
    use rand::{Rng, RngCore, SeedableRng};

    struct TestReader<'a, S>
    where
        S: AsRef<str>,
    {
        stream_input: &'a [S],
    }

    struct StreamExpectation {
        line: String,
        result: Vec<CallBackResult>,
    }

    #[test]
    fn test_marsdecx() {
        let marsdecx = marsdecx();
        let stream_input = marsdecx
            .iter()
            .map(|expectation| expectation.line.as_str())
            .collect::<Vec<_>>();
        let expected_result = marsdecx
            .iter()
            .flat_map(|expectation| expectation.result.clone())
            .collect::<Vec<_>>();
        test_marsdec_core::<true>(&stream_input, &expected_result, None);
    }

    #[test]
    fn test_marsdec1() {
        let marsdec1 = marsdec1();
        let stream_input = marsdec1
            .iter()
            .map(|expectation| expectation.line.as_str())
            .collect::<Vec<_>>();
        let expected_result = marsdec1
            .iter()
            .flat_map(|expectation| expectation.result.clone())
            .collect::<Vec<_>>();
        test_marsdec_core::<false>(&stream_input, &expected_result, None);
    }

    #[test]
    fn man() {
        // Expected:  CallBackResult { line: 1, integer: 7, bit_seq: 0 }
        // Got:       CallBackResult { line: 2, integer: 1, bit_seq: -2 }
        const USE_MARSDECX: bool = true;

        let seed = 566386752197768851;

        let (lines, expected) = generate_test_data::<USE_MARSDECX>(seed);

        test_marsdec_core::<USE_MARSDECX>(&lines, &expected, Some(seed));
    }

    #[test]
    fn test_marsdecx_random() {
        const USE_MARSDECX: bool = true;
        let mut rng = rand::rng();

        let seed = rng.random_range(0..=u64::MAX);

        let (lines, expected) = generate_test_data::<USE_MARSDECX>(seed);

        test_marsdec_core::<USE_MARSDECX>(&lines, &expected, Some(seed));
    }

    #[test]
    fn test_marsdecx_fuzzy() {
        const USE_MARSDECX: bool = true;
        let mut rng = rand::rng();

        for _ in 0..3823 {
            let seed = rng.random_range(0..=u64::MAX);

            let (lines, expected) = generate_test_data::<USE_MARSDECX>(seed);

            test_marsdec_core::<USE_MARSDECX>(&lines, &expected, Some(seed));
        }
    }

    #[test]
    fn test_marsdec1_fuzzy() {
        const USE_MARSDECX: bool = false;
        let mut rng = rand::rng();

        for _ in 0..u8::MAX {
            let seed = rng.random_range(0..=u64::MAX);

            let (lines, expected) = generate_test_data::<USE_MARSDECX>(seed);

            test_marsdec_core::<USE_MARSDECX>(&lines, &expected, Some(seed));
        }
    }

    #[test]
    fn test_marsdec1_random() {
        const USE_MARSDECX: bool = false;
        let mut rng = rand::rng();

        let seed = rng.random_range(0..=u64::MAX);

        let (lines, expected) = generate_test_data::<USE_MARSDECX>(seed);

        test_marsdec_core::<USE_MARSDECX>(&lines, &expected, Some(seed));
    }

    fn generate_test_data<const USE_MARSDECX: bool>(
        seed: u64,
    ) -> (Vec<String>, Vec<CallBackResult>) {
        fn generate_test_data_core<const USE_MARSDECX: bool>(
            rng: &mut StdRng,
            line: u32,
        ) -> (String, Vec<CallBackResult>) {
            let raw_input = ((rng.next_u64() as u128) << 64) | rng.next_u64() as u128;
            let mut _raw_input = raw_input;

            let mut stream_output = Vec::new();

            let mut bit_shift_tally = 0;
            for integer in 1.. {
                let first_5 = ((_raw_input >> (size_of_in_bits::<u128>() - 5)) & 0b11111) as u8;
                _raw_input <<= 5;

                bit_shift_tally += 5;

                if first_5 == 0 {
                    break;
                }

                let next_n = (_raw_input >> (size_of_in_bits::<u128>() - first_5 as usize)) as u32
                    // SAFETY: We know that the maximum number possible from any combination of 5 bits is 31
                    // which is less than 64
                    & unsafe { make_one_bits(first_5) } as u32;

                _raw_input <<= first_5 as usize;
                bit_shift_tally += first_5 as usize;

                // If we have read all the bits, then break
                if bit_shift_tally > size_of_in_bits::<u128>() {
                    break;
                }
                stream_output.push(CallBackResult {
                    line,
                    integer,
                    bit_seq: BitSequence::new(
                        next_n,
                        NonZeroU8::new(first_5).expect("Expected a non-zero value"),
                    )
                    .using_twos_complement(),
                });

                if !USE_MARSDECX {
                    break;
                }
            }

            use std::fmt::Write;
            let byte_stream = raw_input
                .to_be_bytes()
                .iter()
                .fold(String::with_capacity(16), |mut acc, byte| {
                    _ = write!(acc, "{byte:02x} ");
                    acc
                })
                .trim()
                .to_string();

            (byte_stream, stream_output)
        }

        let mut rng = StdRng::seed_from_u64(seed);

        let num_streams = rng.random_range(1..=20);

        (1..num_streams).fold((vec![], vec![]), |(mut input, mut res), line_num| {
            let (i, r) = generate_test_data_core::<USE_MARSDECX>(&mut rng, line_num);
            input.push(i);
            res.extend(r);
            (input, res)
        })
    }

    fn test_marsdec_core<const DECODE_MULTIPLE: bool>(
        stream_input: &[impl AsRef<str>],
        mut expected_result: &[CallBackResult],
        seed: Option<u64>,
    ) {
        let mut reader = TestReader { stream_input };

        Assignment2::marsdec_core::<DECODE_MULTIPLE>(&mut reader, |result| {
            expected_result
                .split_first()
                .map(|(expected, rest)| {
                    expected_result = rest;
                    assert_eq!(
                        result, *expected,
                        "Using Seed {:?}\nExpected:  {:?}\nGot:       {:?}",
                        seed, expected, result
                    );
                })
                .expect("More results than expected");
        });

        assert_eq!(
            expected_result.len(),
            0,
            "Not all results were processed {:?} were not checked",
            expected_result
        );
    }

    fn marsdecx() -> Vec<StreamExpectation> {
        vec![
            StreamExpectation {
                line: "0x08 0x30".to_string(),
                result: vec![
                    CallBackResult {
                        line: 1,
                        integer: 1,
                        bit_seq: 0,
                    },
                    CallBackResult {
                        line: 1,
                        integer: 2,
                        bit_seq: -1,
                    },
                ],
            },
            StreamExpectation {
                line: "0x12 0x28 0x6c 0x70".to_string(),
                result: vec![
                    CallBackResult {
                        line: 2,
                        integer: 1,
                        bit_seq: 1,
                    },
                    CallBackResult {
                        line: 2,
                        integer: 2,
                        bit_seq: -2,
                    },
                    CallBackResult {
                        line: 2,
                        integer: 3,
                        bit_seq: 3,
                    },
                    CallBackResult {
                        line: 2,
                        integer: 4,
                        bit_seq: -4,
                    },
                ],
            },
            StreamExpectation {
                line: "0x43 0xdb 0x13 0x4b 0x2b 0xc6 0x14 0xe0".to_string(),
                result: vec![
                    CallBackResult {
                        line: 3,
                        integer: 1,
                        bit_seq: 123,
                    },
                    CallBackResult {
                        line: 3,
                        integer: 2,
                        bit_seq: 1234,
                    },
                    CallBackResult {
                        line: 3,
                        integer: 3,
                        bit_seq: 12345678,
                    },
                ],
            },
        ]
    }

    fn marsdec1() -> Vec<StreamExpectation> {
        vec![
            StreamExpectation {
                line: "0x08".to_string(),
                result: vec![CallBackResult {
                    line: 1,
                    integer: 1,
                    bit_seq: 0,
                }],
            },
            StreamExpectation {
                line: "0x0c".to_string(),
                result: vec![CallBackResult {
                    line: 2,
                    integer: 1,
                    bit_seq: -1,
                }],
            },
            StreamExpectation {
                line: "0x12".to_string(),
                result: vec![CallBackResult {
                    line: 3,
                    integer: 1,
                    bit_seq: 1,
                }],
            },
            StreamExpectation {
                line: "0x43 0xd8".to_string(),
                result: vec![CallBackResult {
                    line: 4,
                    integer: 1,
                    bit_seq: 123,
                }],
            },
            StreamExpectation {
                line: "0x62 0x69 0x00".to_string(),
                result: vec![CallBackResult {
                    line: 5,
                    integer: 1,
                    bit_seq: 1234,
                }],
            },
            StreamExpectation {
                line: "0xca 0xf1 0x85 0x38".to_string(),
                result: vec![CallBackResult {
                    line: 6,
                    integer: 1,
                    bit_seq: 12345678,
                }],
            },
            StreamExpectation {
                line: "0xfb 0xff 0xff 0xff 0xf0".to_string(),
                result: vec![CallBackResult {
                    line: 7,
                    integer: 1,
                    bit_seq: 1073741823,
                }],
            },
            StreamExpectation {
                line: "0xfc 0x00 0x00 0x00 0x10".to_string(),
                result: vec![CallBackResult {
                    line: 8,
                    integer: 1,
                    bit_seq: -1073741823,
                }],
            },
        ]
    }

    impl<S: AsRef<str>> StreamReader for TestReader<'_, S> {
        type BStream<'stream>
            = HexStream<'stream>
        where
            Self: 'stream;

        fn next_stream(&mut self) -> Option<Self::BStream<'_>> {
            self.stream_input
                .split_first()
                .map(|(raw_stream, streams)| {
                    self.stream_input = streams;
                    HexStream::new(raw_stream.as_ref().split_whitespace())
                })
        }
    }
}
