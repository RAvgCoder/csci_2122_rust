/// Returns the size of a type `T` in bits.
#[inline(always)]
pub const fn size_of_in_bits<T>() -> usize {
    size_of::<T>() * 8
}

/// Returns a number where the last `n` bits are set to 1.
///
/// # Safety
/// The caller must ensure that the number of bits to create `n` must be less than 64.
///
/// # Arguments
///
/// * `n` - The number of bits to set to 1.
///
/// # Examples
///
/// ```
/// unsafe {
///     let bits = make_one_bits(5);
///     assert_eq!(bits, 0b11111);
/// }
/// ```
#[inline(always)]
#[warn(dead_code)]
pub const unsafe fn make_one_bits(n: u8) -> u64 {
    (1 << n) - 1
}

pub fn read_line(std_in: &mut std::io::Stdin) -> Result<String, std::io::Error> {
    let mut buffer = String::new();
    std_in.read_line(&mut buffer).map(|_| buffer)
}
