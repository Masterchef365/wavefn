/// https://en.wikipedia.org/wiki/Permuted_congruential_generator
pub struct Rng {
    state: u64, 
    multiplier: u64, 
    increment: u64,
}

impl Rng {
    pub fn new() -> Self {
        Self::from_seed(
            5573589319906701683,
            6364136223846793005,
            1442695040888963407,
        )
    }

    pub fn from_seed(seed: u64, multiplier: u64, increment: u64) -> Self {
        Self { 
            state: seed + increment,
            multiplier,
            increment,
        }
    }

    fn u64_to_u32(x: u64) -> u32 {
        let bytes = x.to_le_bytes();
        u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[2]])
    }

    fn rotr32(x: u32, r: u32) -> u32 {
        x >> r | x << (r.wrapping_neg() & 31)
    }

    pub fn gen(&mut self) -> u32 {
        let mut x = self.state;
        let count = x >> 59;
        self.state = x.wrapping_mul(self.multiplier).wrapping_add(self.increment);
        x ^= x >> 18;
        Self::rotr32(Self::u64_to_u32(x >> 27), Self::u64_to_u32(count))
    }
}
