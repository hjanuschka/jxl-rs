#![no_main]

use jxl::{
    bit_reader::BitReader,
    encode::BitWriter,
};
use libfuzzer_sys::fuzz_target;

fn mask(bits: usize) -> u64 {
    if bits == 0 { 0 } else { (1u64 << bits) - 1 }
}

fuzz_target!(|data: &[u8]| {
    let mut writer = BitWriter::new();
    let mut expected = Vec::new();

    for chunk in data.chunks_exact(9) {
        let bits = (chunk[0] as usize) % 57;
        let value = u64::from_le_bytes(chunk[1..9].try_into().unwrap());

        writer.write(bits, value).unwrap();
        expected.push((bits, value & mask(bits)));
    }

    let total_bits = writer.total_bits_written();
    let bytes = writer.finish();

    let mut reader = BitReader::new(&bytes);
    for (bits, value) in expected {
        let got = reader.read(bits).unwrap();
        assert_eq!(got, value);
    }

    let pad_bits = (8 - (total_bits % 8)) % 8;
    if pad_bits > 0 {
        assert_eq!(reader.read(pad_bits).unwrap(), 0);
    }
});
