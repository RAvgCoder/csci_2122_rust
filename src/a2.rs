use crate::a2::stream_core::{BitStream, ConsoleReader, StreamReader};
use std::num::NonZeroU8;

pub struct Assignment2;

struct CallBackResult {
    line: i32,
    integer: i32,
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

            let mut bit_stream = BitStream::new(&mut hex_stream);
            for j in 1.. {
                const HEADER_SIZE: NonZeroU8 = match NonZeroU8::new(5) {
                    None => panic!("Expected a non-zero value"),
                    Some(val) => val,
                };

                let Some(bit_seq) = bit_stream.take(HEADER_SIZE) else {
                    break;
                };

                // If the first 5 bits are 0, then we have reached the end of the stream
                let Some(first_5_bits_value) = NonZeroU8::new(bit_seq.get() as u8) else {
                    break;
                };

                let Some(bit_seq) = bit_stream.take(first_5_bits_value) else {
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
    use std::fmt::Debug;
    use std::num::NonZeroU8;

    pub const fn size_of_in_bits<T>() -> usize {
        size_of::<T>() * 8
    }

    /// Returns a number where the last n bits are set to 1.
    ///
    /// # Safety
    /// The caller must ensure that the number of bits to create `n`, must be less than 64.
    #[inline(always)]
    const fn make_one_bits(n: u8) -> u64 {
        assert!(n <= 64, "Cannot make more than 64 bits");
        (1 << n) - 1
    }

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
        fn new(seq: u32, len: NonZeroU8) -> Self {
            BitSequence { seq, len }
        }

        pub fn get(&self) -> u32 {
            self.seq
        }

        pub fn using_twos_complement(&self) -> i32 {
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

    pub struct BitStream<'stream, S> {
        /// The stream that provides the bytes to be read
        byte_stream: &'stream mut S,
        /// The cache that we would use to store the bits that we have read
        cache: u64,
        /// The number of bits in the cache
        cache_size: u8,
    }

    impl<'stream, S: ByteStream> BitStream<'stream, S> {
        /// The maximum number of bits that can be stored in the cache
        const MAX_CACHE_SIZE: u8 = size_of_in_bits::<u64>() as u8;

        pub fn new(byte_stream: &'stream mut S) -> Self {
            BitStream {
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

            let result_mask = make_one_bits(num_bits.get());
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

    pub struct HexStream<'reader>(std::str::SplitWhitespace<'reader>);

    impl HexStream<'_> {
        #[inline(always)]
        pub fn new(iter: std::str::SplitWhitespace) -> HexStream {
            HexStream(iter)
        }
    }

    impl<'reader> ByteStream for HexStream<'reader> {
        fn next_byte(&mut self) -> Option<u8> {
            self.0.next().map(|hex| {
                let stripped_hex = hex.strip_prefix("0x").unwrap_or(hex);
                u8::from_str_radix(stripped_hex, 16).expect("Expected a valid hex byte")
            })
        }
    }

    pub trait StreamReader {
        type BStream<'stream>: ByteStream
        where
            Self: 'stream;
        fn next_stream(&mut self) -> Option<Self::BStream<'_>>;
    }

    /// Reads byte streamline by line from std-in
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

        /// 4 0x12 0x28 0x6c 0x70
        /// Produces the next
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
                .trim()
                .split_whitespace();

            let _ = iter.next();

            Some(HexStream::new(iter))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::a2::stream_core::HexStream;

    struct TestReader<'a> {
        streams: &'a [&'a str],
    }

    const MARSDEC1_EXAMPLE: ([&str; 8], [CallBackResult; 8]) = (
        [
            "0x08",
            "0x0c",
            "0x12",
            "0x43 0xd8",
            "0x62 0x69 0x00",
            "0xca 0xf1 0x85 0x38",
            "0xfb 0xff 0xff 0xff 0xf0",
            "0xfc 0x00 0x00 0x00 0x10",
        ],
        [
            CallBackResult {
                line: 1,
                integer: 1,
                bit_seq: 0,
            },
            CallBackResult {
                line: 2,
                integer: 1,
                bit_seq: -1,
            },
            CallBackResult {
                line: 3,
                integer: 1,
                bit_seq: 1,
            },
            CallBackResult {
                line: 4,
                integer: 1,
                bit_seq: 123,
            },
            CallBackResult {
                line: 5,
                integer: 1,
                bit_seq: 1234,
            },
            CallBackResult {
                line: 6,
                integer: 1,
                bit_seq: 12345678,
            },
            CallBackResult {
                line: 7,
                integer: 1,
                bit_seq: 1073741823,
            },
            CallBackResult {
                line: 8,
                integer: 1,
                bit_seq: -1073741823,
            },
        ],
    );

    const MARSDECX_EXAMPLE: ([&str; 3], [CallBackResult; 9]) = (
        [
            "0x08 0x30",
            "0x12 0x28 0x6c 0x70",
            "0x43 0xdb 0x13 0x4b 0x2b 0xc6 0x14 0xe0",
        ],
        [
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
    );

    #[test]
    fn test_marsdecx() {
        test_marsdec_core::<true>(&MARSDECX_EXAMPLE.0, &MARSDECX_EXAMPLE.1);
    }

    #[test]
    fn test_marsdec1() {
        test_marsdec_core::<false>(&MARSDEC1_EXAMPLE.0, &MARSDEC1_EXAMPLE.1);
    }

    fn test_marsdec_core<const DECODE_MULTIPLE: bool>(
        test: &[&str],
        mut expected_result: &[CallBackResult],
    ) {
        let mut reader = TestReader { streams: test };

        Assignment2::marsdec_core::<DECODE_MULTIPLE>(&mut reader, |result| {
            expected_result.split_first().map(|(expected, rest)| {
                expected_result = rest;
                assert_eq!(result.line, expected.line);
                assert_eq!(result.integer, expected.integer);
                assert_eq!(result.bit_seq, expected.bit_seq);
            });
        });
    }

    impl StreamReader for TestReader<'_> {
        type BStream<'stream>
            = HexStream<'stream>
        where
            Self: 'stream;

        fn next_stream(&mut self) -> Option<Self::BStream<'_>> {
            self.streams.split_first().map(|(raw_stream, streams)| {
                self.streams = streams;
                HexStream::new(raw_stream.split_whitespace())
            })
        }
    }
}
